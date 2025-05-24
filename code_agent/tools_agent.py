import pandas as pd
from smolagents import Tool
from typing import Any, Dict, Optional

class ReverseTextTool(Tool):
    name = "reverse_text"
    description = "Reverses the input text."
    # tell the validator: I’m expecting a dict with key "text"
    inputs = {"input": {"type": "any", "description": "The text to be reversed"}}
    output_type = "string"

    def forward(self, input: Any) -> Any:
        return input[::-1]


class TableCommutativityTool(Tool):
    name = "find_non_commutative_elements"
    description = (
        "Given a multiplication table (2D list) and its header elements, "
        "returns the elements involved in any a*b != b*a."
    )
    inputs = {
        "input": {
            "type": "any",
            "description": "Dict with keys 'table' (list of lists) and 'elements' (list of strings)."
        }
    }
    output_type = "string"

    def forward(self, input: dict) -> list[str]:
        table   = input["table"]
        elements = input["elements"]
        non_comm = set()
        for i, a in enumerate(elements):
            for j, b in enumerate(elements):
                if table[i][j] != table[j][i]:
                    non_comm.update({a, b})
        return str(sorted(non_comm))



class VegetableListTool(Tool):
    name = "list_vegetables"
    description = (
        "From a list of grocery items, returns those that are true vegetables "
        "(botanical definition), sorted alphabetically."
    )
    inputs = {
        "input": {
            "type": "any",
            "description": "Dict with key 'items' containing a list of item strings."
        }
    }
    output_type = "string"

    _VEG_SET = {
        "broccoli", "bell pepper", "celery", "corn",
        "green beans", "lettuce", "sweet potatoes", "zucchini"
    }

    def forward(self, input: Any) -> Any:
        items = input["items"]
        return str(sorted(item for item in items if item in self._VEG_SET))


class ExcelSumFoodTool(Tool):
    name = "sum_food_sales"
    description = (
        "Reads an Excel file with columns 'Category' and 'Sales', "
        "and returns total sales where Category != 'Drink', rounded to two decimals."
    )
    inputs = {
        "input": {
            "type": "any",
            "description": "Dict with key 'excel_path' pointing to the .xlsx file to read."
        }
    }
    output_type = "string"

    def forward(self, input: Any) -> Any:
        excel_path = input["excel_path"]
        df = pd.read_excel(excel_path)
        total = df.loc[df["Category"] != "Drink", "Sales"].sum()
        return str(round(float(total), 2))