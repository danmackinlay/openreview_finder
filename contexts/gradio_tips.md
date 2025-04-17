To display dataframes in Gradio UI, you use the `gr.Dataframe` component, which can show pandas DataFrames, numpy arrays, lists, or matrices in a spreadsheet-like table. This component can be used both as an input (for users to edit or upload data) or as an output (to display results).

## Basic Usage

Here's a simple example of displaying a pandas DataFrame as output in a Gradio interface:

```python
import gradio as gr
import pandas as pd

def get_data():
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

demo = gr.Interface(
    fn=get_data,
    inputs=None,
    outputs=gr.Dataframe(headers=["A", "B"])
)

demo.launch()
```
This will display the DataFrame in the Gradio UI when the interface is launched[1][3].

## Customization Options

- **Headers and Data Types**: You can specify column headers and data types using the `headers` and `datatype` parameters.
- **Row and Column Count**: Use `row_count` and `col_count` to set the number of rows and columns.
- **Interactivity**: Set `interactive=True` to allow users to edit the table, or `interactive=False` to make it read-only[4].
- **Fixed Height and Scrolling**: You can set a fixed height for the DataFrame to enable scrolling if the content exceeds the visible area. This is useful for large tables[4].
- **Search and Filter**: The `show_search` parameter adds a search/filter bar above the table for easier navigation[1].

## Styling DataFrames

Gradio supports pandas' `Styler` objects for advanced styling, such as highlighting cells, changing text color, or applying gradients:

```python
import pandas as pd
import gradio as gr

df = pd.DataFrame({
    "A": [14, 4, 5, 4, 1],
    "B": [5, 2, 54, 3, 2]
})

# Highlight the maximum value in each column
styler = df.style.highlight_max(color='lightgreen', axis=0)

with gr.Blocks() as demo:
    gr.Dataframe(styler)
demo.launch()
```
This will display a styled DataFrame with colored highlights[2].

## Example: DataFrame as Output Based on User Input

```python
import gradio as gr
import pandas as pd

def filter_records(records, gender):
    return records[records["gender"] == gender]

demo = gr.Interface(
    filter_records,
    [
        gr.Dataframe(
            headers=["name", "age", "gender"],
            datatype=["str", "number", "str"],
            row_count=5,
            col_count=(3, "fixed"),
        ),
        gr.Dropdown(["M", "F", "O"]),
    ],
    "dataframe",
    description="Enter gender as 'M', 'F', or 'O' for other.",
)

demo.launch()
```
This interface takes a DataFrame and a dropdown selection as input and displays a filtered DataFrame as output[1].

## Notes

- For large DataFrames, consider using the `height` parameter for better UI control and scrollability[4].
- You can use custom CSS or pandas `Styler` for advanced visual customization[2].

Gradio's `Dataframe` component is flexible and powerful for displaying and interacting with tabular data in web apps.

Citations:
[1] https://www.gradio.app/docs/gradio/dataframe
[2] https://www.gradio.app/guides/styling-the-gradio-dataframe
[3] https://discuss.huggingface.co/t/how-to-use-gradio-dataframe-as-output-for-an-interface/19798
[4] https://discuss.huggingface.co/t/the-output-of-dataframe/50380
[5] https://www.reddit.com/r/nicegui/comments/11qzucj/display_pandas_dataframes/
[6] https://github.com/gradio-app/gradio/issues/7077
[7] https://github.com/gradio-app/gradio/issues/7423
[8] https://colab.research.google.com
