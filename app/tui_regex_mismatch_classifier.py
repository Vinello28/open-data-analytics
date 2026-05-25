import os
import re
import time
import socket
import asyncio
import json

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
import questionary
from openai import AsyncOpenAI


console = Console()


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


async def check_port(ip, port, timeout=0.2):
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return ip
    except Exception:
        return None


async def discover_servers(port=1234):
    local_ip = get_local_ip()
    if local_ip == "127.0.0.1":
        return ["127.0.0.1"]

    parts = local_ip.split(".")
    base_ip = f"{parts[0]}.{parts[1]}.{parts[2]}."

    tasks = []
    for i in range(1, 255):
        tasks.append(check_port(base_ip + str(i), port))

    results = await asyncio.gather(*tasks)
    found_ips = [ip for ip in results if ip is not None]

    if "127.0.0.1" not in found_ips and await check_port("127.0.0.1", port):
        found_ips.append("127.0.0.1")

    return found_ips


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_data():
    project_root = get_project_root()
    input_path = os.path.join(project_root, "data", "regex_tp_matches.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str)
    expected_columns = {"old", "descrizione"}
    missing = expected_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {input_path}: {sorted(missing)}")

    df = df.fillna("")
    records = []
    for idx, row in df.iterrows():
        records.append({
            "row_index": idx,
            "old": str(row["old"]).strip(),
            "text": str(row["descrizione"]).strip(),
        })
    return records, input_path


def normalize_label_candidate(value):
    normalized = re.sub(r"[^a-z]+", "", str(value).strip().lower())
    if normalized == "nonai":
        return "non_ai"
    if normalized == "ai":
        return "ai"
    return None


def iter_json_candidates(value):
    if isinstance(value, dict):
        for key in ("label", "classification", "predicted_label", "prediction", "class"):
            if key in value:
                yield value[key]
        for nested_value in value.values():
            yield from iter_json_candidates(nested_value)
    elif isinstance(value, list):
        for item in value:
            yield from iter_json_candidates(item)
    else:
        yield value


def extract_label(content):
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    if "{" in cleaned and "}" in cleaned:
        json_candidate = cleaned[cleaned.find("{"):cleaned.rfind("}") + 1]
        try:
            parsed = json.loads(json_candidate)
            for candidate in iter_json_candidates(parsed):
                normalized = normalize_label_candidate(candidate)
                if normalized:
                    return normalized
        except json.JSONDecodeError:
            pass

    label_match = re.search(
        r"(?is)\b(?:label|classification|prediction|predicted_label|class)\b\s*[:=]\s*[\"']?([a-zA-Z_\-\s]+)[\"']?",
        cleaned,
    )
    if label_match:
        normalized = normalize_label_candidate(label_match.group(1))
        if normalized:
            return normalized

    direct_match = re.search(r"(?i)\b(non[-_\s]?ai|ai)\b", cleaned)
    if direct_match:
        normalized = normalize_label_candidate(direct_match.group(1))
        if normalized:
            return normalized

    return "error: unknown_label"


async def classify_text(client, text, model_name):
    prompt = f"""Classifica il seguente testo in una sola etichetta: 'ai' oppure 'non_ai'.
Pensa internamente quanto serve, ma non mostrare il ragionamento.
Rispondi esclusivamente con JSON valido nel formato: {{"label": "ai"}} oppure {{"label": "non_ai"}}.
Non aggiungere testo extra, spiegazioni, markdown o blocchi di codice.

Testo:
{text}
"""

    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a classifier that returns only structured JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["ai", "non_ai"]
                    }
                },
                "required": ["label"],
                "additionalProperties": False
            }
        }
    }

    try:
        try:
            response = await client.chat.completions.create(
                **request_kwargs,
                response_format=schema,
            )
        except Exception as json_error:
            try:
                response = await client.chat.completions.create(
                    **request_kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = await client.chat.completions.create(**request_kwargs)

        content = response.choices[0].message.content or ""
        return extract_label(content)
    except Exception as e:
        return f"error: {str(e)}"


async def process_batch(client, data, model_name, concurrency_limit):
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Classifying mismatch descriptions...", total=len(data))
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def bounded_classify(item):
            async with semaphore:
                pred = await classify_text(client, item["text"], model_name)
                item["predicted_label"] = pred
                progress.advance(task)
                return item

        tasks = [bounded_classify(item) for item in data]
        results = await asyncio.gather(*tasks)

    return results


def build_model_support_table(available_models, requested_model):
    table = Table(title="Available Models")
    table.add_column("Status", style="cyan")
    table.add_column("Model", style="magenta")

    requested_lower = requested_model.lower()
    matched = False
    for model_id in available_models[:20]:
        is_match = requested_lower in model_id.lower() or "gpt-oss-20b" in model_id.lower()
        if is_match:
            matched = True
        table.add_row("✓" if is_match else " ", model_id)

    return table, matched


async def main():
    console.print(Panel.fit("[bold magenta]🧪 Regex FN Reclassifier[/bold magenta]\n[dim]LM Studio / OpenAI-compatible endpoint[/dim]"))

    with console.status("[bold green]🔍 Scanning local network for LM Studio instances (port 1234)..."):
        found_servers = await discover_servers()

    choices = [f"http://{ip}:1234/v1" for ip in found_servers]
    choices.append("Manual Input")

    server_choice = await questionary.select(
        "Select LM Studio server endpoint:",
        choices=choices,
    ).ask_async()

    if not server_choice:
        console.print("[red]Operation cancelled.[/red]")
        return

    if server_choice == "Manual Input":
        server_choice = await questionary.text(
            "Enter server endpoint (e.g., http://192.168.1.100:1234/v1):",
            default="http://localhost:1234/v1",
        ).ask_async()

    if not server_choice:
        return

    model_name = await questionary.text(
        "Enter model name:",
        default="gpt-oss-20b",
    ).ask_async()

    if not model_name:
        console.print("[red]Operation cancelled.[/red]")
        return

    console.print(f"[green]✓[/green] Server selected: [bold]{server_choice}[/bold]")
    console.print(f"[green]✓[/green] Model selected: [bold]{model_name}[/bold]")

    concurrency_str = await questionary.text(
        "Enter the number of concurrent requests:",
        default="4",
        validate=lambda text: text.isdigit() and int(text) > 0 or "Please enter a valid positive integer",
    ).ask_async()

    if not concurrency_str:
        console.print("[red]Operation cancelled.[/red]")
        return

    concurrency_limit = int(concurrency_str)
    console.print(f"[green]✓[/green] Concurrency limit: [bold]{concurrency_limit}[/bold]")

    confirm = await questionary.confirm("Start classification of regex FN mismatches?").ask_async()
    if not confirm:
        console.print("[yellow]Inference cancelled by user.[/yellow]")
        return

    client = AsyncOpenAI(base_url=server_choice, api_key="lm-studio")

    with console.status("[bold cyan]Testing connection to server..."):
        try:
            models_response = await client.models.list()
            available_models = [getattr(model, "id", str(model)) for model in getattr(models_response, "data", [])]
            console.print("[green]✓[/green] Connection successful.")

            if available_models:
                support_table, matched = build_model_support_table(available_models, model_name)
                console.print(support_table)
                if not matched:
                    console.print("[yellow]Requested model not found in the first server listing, but the run can continue if the endpoint accepts the name.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]❌ Connection failed:[/bold red] {e}")
            console.print("[red]Please ensure LM Studio is running and the local server is started.[/red]")
            return

    console.print("[cyan]Loading mismatch data...[/cyan]")
    try:
        data, input_path = load_data()
    except Exception as e:
        console.print(f"[bold red]❌ Input loading failed:[/bold red] {e}")
        return

    console.print(f"[green]✓[/green] Loaded [bold]{len(data)}[/bold] mismatch descriptions from [bold]{input_path}[/bold].")

    if len(data) == 0:
        console.print("[red]No data found to process.[/red]")
        return

    start_time = time.time()
    results = await process_batch(client, data, model_name, concurrency_limit)
    elapsed = time.time() - start_time

    df = pd.DataFrame(results)
    valid_preds = df[~df["predicted_label"].astype(str).str.startswith("error")]
    errors = df[df["predicted_label"].astype(str).str.startswith("error")]

    if len(valid_preds) > 0:
        ai_count = (valid_preds["predicted_label"] == "ai").sum()
        non_ai_count = (valid_preds["predicted_label"] == "non_ai").sum()
        model_agreement = (valid_preds["predicted_label"].str.lower() == valid_preds["old"].str.lower()).sum()
        agreement_rate = model_agreement / len(valid_preds) * 100
    else:
        ai_count = 0
        non_ai_count = 0
        agreement_rate = 0.0

    items_per_second = len(data) / elapsed if elapsed > 0 else 0

    project_root = get_project_root()
    output_dir = os.path.join(project_root, "data", "labelled")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "regex_tp_reclassified.csv")

    export_df = df[["old", "text", "predicted_label"]].copy()
    export_df.rename(columns={"text": "descrizione"}, inplace=True)
    export_df.to_csv(output_path, index=False)

    console.print("\n")
    metrics_table = Table(title="📊 Performance Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="magenta")

    metrics_table.add_row("Total Processed", str(len(data)))
    metrics_table.add_row("Successful", str(len(valid_preds)))
    metrics_table.add_row("Errors", str(len(errors)))
    metrics_table.add_row("Predicted AI", str(ai_count))
    metrics_table.add_row("Predicted NON_AI", str(non_ai_count))
    metrics_table.add_row("Agreement vs original", f"{agreement_rate:.2f}%")
    metrics_table.add_row("Time Taken", f"{elapsed:.2f} seconds")
    metrics_table.add_row("Speed", f"{items_per_second:.2f} items/sec")

    console.print(metrics_table)
    console.print(f"\n[bold green]✅ Results saved to:[/bold green] {output_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Process interrupted by user.[/red]")