import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


def try_convert_type(value: str) -> Union[int, float, None, str]:
    # Try to convert string to int or float, return None if empty
    if value == '':
        return None
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return value


def _split_csv_line(line: str, sep: str = ',') -> List[str]:
    # Split CSV line handling quotes and escaped quotes
    out = []
    cur = []
    in_quotes = False
    i = 0
    n = len(line)
    
    while i < n:
        ch = line[i]
        if ch == '"':
            if in_quotes and i + 1 < n and line[i + 1] == '"':
                cur.append('"')
                i += 2
                continue
            in_quotes = not in_quotes
            i += 1
            continue
        if ch == sep and not in_quotes:
            out.append(''.join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    
    out.append(''.join(cur))
    return out


def custom_csv_parser(file_path: Union[str, Path], separator: str = ',') -> Dict[str, List[Any]]:
    path = Path(file_path) if not isinstance(file_path, Path) else file_path
    
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")
    
    if path.stat().st_size == 0:
        return {}
    
    data = {}
    with open(path, 'r', newline='') as f:
        header_line = f.readline().rstrip('\r\n')
        headers = _split_csv_line(header_line, separator)
        for h in headers:
            data[h] = []

        for raw in f:
            line = raw.rstrip('\r\n')
            if not line:
                continue
            values = _split_csv_line(line, separator)

            if len(values) < len(headers):
                values += [''] * (len(headers) - len(values))

            if len(values) > len(headers):
                values = values[:len(headers)]

            for i, h in enumerate(headers):
                data[h].append(try_convert_type(values[i]))
    return data

def to_float_or_none(x: Any) -> Optional[float]:
    # Convert to float or return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
