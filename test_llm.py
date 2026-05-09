import asyncio
from openai import AsyncOpenAI
import json

async def main():
    client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    try:
        response = await client.chat.completions.create(
            model="mistralai/ministral-3-3b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that strictly returns valid JSON."},
                {"role": "user", "content": "hello"}
            ],
            temperature=0.0,
            max_tokens=20,
            response_format={ "type": "json_object" }
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {repr(e)}")

asyncio.run(main())
