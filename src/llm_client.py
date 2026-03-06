"""
Centralized LLM client for Azure OpenAI calls.

All modules should use these functions instead of creating their own clients.
"""
import time
import sys
from openai import OpenAI
from tqdm import tqdm

from config import Config


def call_azure_openai(
    messages: list,
    max_tokens: int = 16000,
    temperature: float = 0.3,
    max_retries: int = 3,
    json_mode: bool = True,
) -> str:
    """
    Call Azure OpenAI API with retry logic and rate-limit backoff.

    Args:
        messages: List of message dicts (role, content)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Number of retries on failure
        json_mode: If True, request JSON response format

    Returns:
        Response content string
    """
    try:
        Config.validate()
    except ValueError as e:
        print(f"  Configuration error: {e}")
        sys.exit(1)

    client = OpenAI(
        base_url=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_KEY,
    )

    kwargs = {
        "model": Config.AZURE_OPENAI_DEPLOYMENT,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    with tqdm(total=1, desc="Calling Azure OpenAI", unit="call") as pbar:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                if finish_reason == "length":
                    print(f"\n  Warning: Response truncated due to token limit!")

                pbar.update(1)
                pbar.set_description(f"  Received response ({len(content)} chars)")
                return content

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "RateLimitReached" in err_str
                if attempt < max_retries - 1:
                    wait = 90 if is_rate_limit else 5
                    pbar.set_description(
                        f"  Attempt {attempt + 1}/{max_retries} failed, retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    pbar.set_description("  All retries failed")
                    print(f"\n  Error: {e}")
                    raise


def call_azure_openai_text(
    messages: list,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> str:
    """
    Call Azure OpenAI API for plain text responses (no JSON mode).

    Args:
        messages: List of message dicts (role, content)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Number of retries on failure

    Returns:
        Response content string
    """
    return call_azure_openai(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        json_mode=False,
    )


def call_azure_openai_for_code(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> str:
    """
    Call Azure OpenAI for code generation.
    Strips markdown code fences from response.

    Args:
        system_prompt: System message content
        user_prompt: User message content
        max_retries: Number of retries on failure
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        Generated code string (fences stripped)
    """
    try:
        Config.validate()
    except ValueError as e:
        raise RuntimeError(f"Configuration error: {e}")

    client = OpenAI(
        base_url=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_KEY,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            code = response.choices[0].message.content.strip()

            # Remove markdown code fences if present
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]

            return code.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt + 1} failed, retrying: {e}")
            else:
                raise RuntimeError(f"LLM call failed after {max_retries} attempts: {e}")
