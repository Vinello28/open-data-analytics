import os
import re
import glob
import time
import socket
import asyncio
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
import questionary
from openai import AsyncOpenAI
import json

console = Console()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

async def check_port(ip, port, timeout=0.2):
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(ip, port), timeout=timeout)
        writer.close()
        await writer.wait_closed()
        return ip
    except:
        return None

async def discover_servers(port=1234):
    local_ip = get_local_ip()
    if local_ip == '127.0.0.1':
        return ['127.0.0.1']
    
    parts = local_ip.split('.')
    base_ip = f"{parts[0]}.{parts[1]}.{parts[2]}."
    
    tasks = []
    # Scan common range to be fast
    for i in range(1, 255):
        ip = base_ip + str(i)
        tasks.append(check_port(ip, port))
    
    results = await asyncio.gather(*tasks)
    found_ips = [ip for ip in results if ip is not None]
    
    # Also check localhost if not in range
    if '127.0.0.1' not in found_ips:
        if await check_port('127.0.0.1', port):
            found_ips.append('127.0.0.1')
            
    return found_ips

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_data():
    project_root = get_project_root()
    ai_files = glob.glob(os.path.join(project_root, "data/extracted_descriptions/ai/*.txt"))
    non_ai_files = glob.glob(os.path.join(project_root, "data/extracted_descriptions/non_ai/*.txt"))
    
    data = []
    for f in ai_files:
        with open(f, 'r', encoding='utf-8') as file:
            data.append({"filename": os.path.basename(f), "text": file.read().strip(), "original_label": "ai"})
            
    for f in non_ai_files:
        with open(f, 'r', encoding='utf-8') as file:
            data.append({"filename": os.path.basename(f), "text": file.read().strip(), "original_label": "non_ai"})
            
    return data

async def classify_text(client, text, model_name):
    prompt = f"""You are an expert classifier. Classify the following project description into one of two categories: 'ai' (if it heavily involves artificial intelligence, machine learning, deep learning, computer vision, agentic ai, llm, ai automation, etc.) or 'non_ai' (if it does not).
Return only a JSON object with a single key 'label' and value either 'ai' or 'non_ai'.

Description:
{text}
"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly returns valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=30
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Try JSON parsing first, but fallback to raw text parsing
        try:
            result = json.loads(content)
            label = str(result.get('label', content)).lower()
        except json.JSONDecodeError:
            label = content.lower()
            
        # Robust regex extraction
        if re.search(r'non[-_\s]?ai', label):
            return 'non_ai'
        elif 'ai' in label:
            return 'ai'
        else:
            return "error: unknown_label"
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
        transient=False
    ) as progress:
        task = progress.add_task("[cyan]Classifying descriptions...", total=len(data))
        
        # Concurrency limit to avoid overloading local LM Studio
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def bounded_classify(item):
            async with semaphore:
                pred = await classify_text(client, item['text'], model_name)
                item['predicted_label'] = pred
                progress.advance(task)
                return item
                
        tasks = [bounded_classify(item) for item in data]
        results = await asyncio.gather(*tasks)
        
    return results

async def main():
    console.print(Panel.fit("[bold magenta]🚀 AI/Non-AI Description Classifier[/bold magenta]\n[dim]Powered by LM Studio[/dim]"))
    
    with console.status("[bold green]🔍 Scanning local network for LM Studio instances (port 1234)..."):
        found_servers = await discover_servers()
        
    choices = [f"http://{ip}:1234/v1" for ip in found_servers]
    choices.append("Manual Input")
    
    server_choice = await questionary.select(
        "Select LM Studio server endpoint:",
        choices=choices
    ).ask_async()
    
    if not server_choice:
        console.print("[red]Operation cancelled.[/red]")
        return
        
    if server_choice == "Manual Input":
        server_choice = await questionary.text(
            "Enter server endpoint (e.g., http://192.168.1.100:1234/v1):",
            default="http://localhost:1234/v1"
        ).ask_async()
        
    if not server_choice:
        return

    model_name = "mistralai/ministral-3-3b" #"qwen/qwen3-4b-2507" #"mistralai/ministral-3-3b"
    
    console.print(f"[green]✓[/green] Server selected: [bold]{server_choice}[/bold]")
    console.print(f"[green]✓[/green] Model selected: [bold]{model_name}[/bold]")
    
    concurrency_str = await questionary.text(
        "Enter the number of concurrent requests:",
        default="4",
        validate=lambda text: text.isdigit() and int(text) > 0 or "Please enter a valid positive integer"
    ).ask_async()
    
    if not concurrency_str:
        console.print("[red]Operation cancelled.[/red]")
        return
        
    concurrency_limit = int(concurrency_str)
    console.print(f"[green]✓[/green] Concurrency limit: [bold]{concurrency_limit}[/bold]")
    
    confirm = await questionary.confirm("Start inference process?").ask_async()
    if not confirm:
        console.print("[yellow]Inference cancelled by user.[/yellow]")
        return
        
    client = AsyncOpenAI(base_url=server_choice, api_key="lm-studio")
    
    # Test connection
    with console.status("[bold cyan]Testing connection to server..."):
        try:
            await client.models.list()
            console.print("[green]✓[/green] Connection successful.")
        except Exception as e:
            console.print(f"[bold red]❌ Connection failed:[/bold red] {e}")
            console.print("[red]Please ensure LM Studio is running and the local server is started.[/red]")
            return
            
    console.print("[cyan]Loading data...[/cyan]")
    data = load_data()
    console.print(f"[green]✓[/green] Loaded [bold]{len(data)}[/bold] descriptions.")
    
    if len(data) == 0:
        console.print("[red]No data found to process.[/red]")
        return
        
    start_time = time.time()
    
    # Process
    results = await process_batch(client, data, model_name, concurrency_limit)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Calculate metrics
    df = pd.DataFrame(results)
    
    valid_preds = df[~df['predicted_label'].str.startswith('error')]
    errors = df[df['predicted_label'].str.startswith('error')]
    
    if len(valid_preds) > 0:
        correct = (valid_preds['original_label'] == valid_preds['predicted_label']).sum()
        accuracy = correct / len(valid_preds) * 100
    else:
        accuracy = 0.0
        
    items_per_second = len(data) / elapsed if elapsed > 0 else 0
    
    # Save results
    project_root = get_project_root()
    output_dir = os.path.join(project_root, "data/labelled")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lm_studio_classification.csv")
    df.to_csv(output_path, index=False)
    
    # Display metrics
    console.print("\n")
    metrics_table = Table(title="📊 Performance Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="magenta")
    
    metrics_table.add_row("Total Processed", str(len(data)))
    metrics_table.add_row("Successful", str(len(valid_preds)))
    metrics_table.add_row("Errors", str(len(errors)))
    if len(errors) > 0:
        first_error = errors.iloc[0]['predicted_label']
        metrics_table.add_row("First Error", str(first_error))
    metrics_table.add_row("Accuracy (vs original)", f"{accuracy:.2f}%")
    metrics_table.add_row("Time Taken", f"{elapsed:.2f} seconds")
    metrics_table.add_row("Speed", f"{items_per_second:.2f} items/sec")
    
    console.print(metrics_table)
    console.print(f"\n[bold green]✅ Results saved to:[/bold green] {output_path}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[red]Process interrupted by user.[/red]")
