import re
import json

def parse_json_robust(text: str) -> dict:
    """Robustly extract and parse a JSON object from text, handling markdown wrappers."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try finding JSON block in markdown ```json ... ```
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
            
    # Try finding anything between the first '{' and the last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"Could not parse valid JSON from LLM response: {text}")
