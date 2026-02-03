import math
from typing import List, Dict, Any, Tuple, Optional, Callable, Union


class GroupBy:
    def __init__(self, df: 'DataFrame', keys: List[str]):
        if df._num_rows == 0:
            raise ValueError("Cannot create GroupBy from empty DataFrame.")
        
        if not keys:
            raise ValueError("Must provide at least one key for grouping.")
        
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise KeyError(f"GroupBy keys not found: {missing}. Available: {df.columns}")
        
        self.df = df
        self.keys = keys
        self.groups = {}
        
        n = df._num_rows
        key_cols = [df._data[k] for k in keys]
        
        for i in range(n):
            kt = tuple(col[i] for col in key_cols)
            self.groups.setdefault(kt, []).append(i)
    
    def agg(self, spec: Dict[str, List[str]]) -> 'DataFrame':
        out_cols = {k: [] for k in self.keys}
        agg_cols = {}
        
        for val_col, funs in spec.items():
            for fn in funs:
                name = f"{fn}_{val_col}"
                agg_cols[name] = []
        
        for kt, idxs in self.groups.items():
            for j, k in enumerate(self.keys):
                out_cols[k].append(kt[j])
            
            for val_col, funs in spec.items():
                if val_col not in self.df._data:
                    for fn in funs:
                        agg_cols[f"{fn}_{val_col}"].append(None)
                    continue
                
                vals = [self.df._data[val_col][i] for i in idxs]
                nums = [v for v in vals if isinstance(v, (int, float)) and v is not None]
                
                for fn in funs:
                    col_name = f"{fn}_{val_col}"
                    
                    if fn == 'count':
                        agg_cols[col_name].append(len(idxs))
                    elif not nums:
                        agg_cols[col_name].append(None)
                    elif fn == 'sum':
                        agg_cols[col_name].append(sum(nums))
                    elif fn == 'avg':
                        if len(nums) == 0:
                            agg_cols[col_name].append(None)
                        else:
                            agg_cols[col_name].append(sum(nums) / len(nums))
                    elif fn == 'min':
                        agg_cols[col_name].append(min(nums))
                    elif fn == 'max':
                        agg_cols[col_name].append(max(nums))
                    elif fn == 'median':
                        sorted_nums = sorted(nums)
                        n = len(sorted_nums)
                        mid = n // 2
                        if n % 2 == 0:
                            agg_cols[col_name].append((sorted_nums[mid - 1] + sorted_nums[mid]) / 2)
                        else:
                            agg_cols[col_name].append(sorted_nums[mid])
                    elif fn == 'std':
                        if len(nums) < 2:
                            agg_cols[col_name].append(None)
                        else:
                            mean = sum(nums) / len(nums)
                            variance = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
                            agg_cols[col_name].append(variance ** 0.5)
                    else:
                        raise ValueError(f"Unsupported aggregation function: {fn}")
        
        out_cols.update(agg_cols)
        return DataFrame(out_cols)


class DataFrame:
    def __init__(self, data: Dict[str, List[Any]]):
        if not isinstance(data, dict):
            raise TypeError(f"Input must be a dictionary, got {type(data).__name__}")
        
        self._data = data
        self._length = len(next(iter(data.values()))) if data else 0
        self._num_rows = self._length
        self._num_cols = len(self._data) if self._data else 0
        
        if data:
            if not all(isinstance(v, list) for v in data.values()):
                raise TypeError("Input data must be a dictionary of lists.")
            if not all(len(v) == self._length for v in data.values()):
                raise ValueError(f"All lists must have the same length. Found lengths: {[len(v) for v in data.values()]}")

    @property
    def columns(self) -> List[str]:
        return list(self._data.keys())

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._length, len(self.columns))

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return f"<DataFrame: {self._num_rows:,} rows x {self._num_cols} columns>"

    def __str__(self) -> str:
        if not self._data:
            return "Empty DataFrame"
        headers = list(self.columns)[:5]
        rows = []
        for i in range(min(5, self._num_rows)):
            row = [str(self._data[h][i])[:15] for h in headers]
            rows.append(" | ".join(row))
        return "Columns: " + ", ".join(headers) + "\n" + "\n".join(rows)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self._data:
                return self._data[item]
            raise KeyError(f"Column '{item}' not found")
        elif isinstance(item, list):
            return self.select(item)
        raise TypeError("Invalid argument type. Use string for single column or list for multiple columns.")

    def select(self, columns: List[str]) -> 'DataFrame':
        if not isinstance(columns, list):
            raise TypeError(f"columns must be a list, got {type(columns).__name__}")
        if len(columns) == 0:
            raise ValueError("Cannot select zero columns. Provide at least one column name.")
        
        if self._num_rows == 0:
            return DataFrame({})
        
        new_data = {c: self._data[c][:] for c in columns if c in self._data}
        
        if not new_data:
            raise ValueError(f"None of the requested columns {columns} exist in DataFrame. Available columns: {self.columns}")
        
        return DataFrame(new_data)

    def filter(self, condition: List[bool]) -> 'DataFrame':
        if not isinstance(condition, list):
            raise TypeError(f"condition must be a list, got {type(condition).__name__}")
        
        if len(condition) != self._length:
            raise ValueError(
                f"Condition list length ({len(condition)}) must match DataFrame length ({self._length})."
            )
        
        if not all(isinstance(c, (bool, int)) or c in (True, False, 0, 1) for c in condition):
            raise TypeError("Condition list must contain only boolean values.")
        
        if self._num_rows == 0:
            return DataFrame({})
        
        bool_condition = [bool(c) for c in condition]
        new_data = {col: [self._data[col][i] for i, c in enumerate(bool_condition) if c] for col in self.columns}
        
        return DataFrame(new_data)

    def sort_values(self, by: str, ascending: bool = True) -> 'DataFrame':
        if by not in self._data:
            raise ValueError(f"Column '{by}' not found.")
        
        indices = list(range(self._length))
        zipped_data = sorted(zip(self._data[by], indices), key=lambda x: (x[0] is None, x[0]), reverse=not ascending)
        sorted_indices = [idx for val, idx in zipped_data]
        
        new_data = {col: [self._data[col][i] for i in sorted_indices] for col in self.columns}
        return DataFrame(new_data)

    def groupby(self, keys: Union[str, List[str]]) -> 'GroupBy':
        if self._num_rows == 0:
            raise ValueError("Cannot groupby on empty DataFrame.")
        
        if isinstance(keys, str):
            keys = [keys]
        elif not isinstance(keys, list):
            raise TypeError(f"keys must be a string or list, got {type(keys).__name__}")
        
        if len(keys) == 0:
            raise ValueError("Must provide at least one column to group by.")
        
        missing_keys = [k for k in keys if k not in self.columns]
        if missing_keys:
            raise ValueError(
                f"GroupBy keys not found in DataFrame: {missing_keys}. "
                f"Available columns: {self.columns}"
            )
        
        return GroupBy(self, keys)

    def agg(self, aggregations: Dict[str, Callable[[List[Any]], Any]]) -> 'DataFrame':
        new_data = {}
        for col, func in aggregations.items():
            if col in self.columns:
                new_data[col] = [func(self._data[col])]
        return DataFrame(new_data)

    def join(self, other: 'DataFrame', on: Tuple[str, str], how: str = 'inner') -> 'DataFrame':
        left_key, right_key = on
        
        if left_key not in self.columns:
            raise ValueError(f"Left join key '{left_key}' not found in left DataFrame.")
        if right_key not in other.columns:
            raise ValueError(f"Right join key '{right_key}' not found in right DataFrame.")
        
        if how not in ('inner', 'left'):
            raise NotImplementedError(f"Join type '{how}' not supported. Use 'inner' or 'left'.")
        
        right_map = {}
        for i, rk in enumerate(other._data[right_key]):
            if rk is not None:
                right_map.setdefault(rk, []).append(i)
        
        out = {c: [] for c in self.columns}
        right_prefix = "r_"
        for c in other.columns:
            out[right_prefix + c] = []
        
        for i, lk in enumerate(self._data[left_key]):
            if lk in right_map:
                for j in right_map[lk]:
                    for c in self.columns:
                        out[c].append(self._data[c][i])
                    for c in other.columns:
                        out[right_prefix + c].append(other._data[c][j])
            elif how == 'left':
                for c in self.columns:
                    out[c].append(self._data[c][i])
                for c in other.columns:
                    out[right_prefix + c].append(None)
        
        return DataFrame(out)

def _safe_corr(df: DataFrame, col1: str, col2: str, min_pairs: int = 3) -> Optional[float]:
    # Calculate correlation between two columns
    if col1 not in df._data or col2 not in df._data:
        return None
    
    xs = [x for x, y in zip(df._data[col1], df._data[col2]) if x is not None and y is not None]
    ys = [y for x, y in zip(df._data[col1], df._data[col2]) if x is not None and y is not None]
    
    if len(xs) < min_pairs:
        return None
    
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_x2 = sum(x*x for x in xs)
    sum_y2 = sum(y*y for y in ys)
    sum_xy = sum(x*y for x,y in zip(xs, ys))
    
    try:
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        return numerator / denominator if denominator != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return None