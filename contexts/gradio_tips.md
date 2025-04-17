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
---

The `gr.Dataset` component in Gradio is used to display a collection of data samples, typically presented as a gallery or table within the Gradio interface[1]. It's primarily intended for displaying example inputs for a model or function, allowing users to easily select and load predefined data points[1][3].

## Key Features and Usage

*   **Components Definition**: You define the structure of each data sample by passing a list of Gradio components (or their string names) to the `components` parameter. Supported components include `Textbox`, `Image`, `Audio`, `Dataframe`, `Dropdown`, and others[1][3].
*   **Providing Samples**: The actual data is provided as a list of lists to the `samples` parameter. Each inner list corresponds to one data sample, with elements matching the order and type of the specified `components`[1][3].
*   **Display**: It renders the samples in a table or gallery format within the UI.
*   **Interaction**: Users can click on a sample in the `Dataset`. This selection can trigger event listeners (`.click()`, `.select()`)[3]. When a sample is selected, the `Dataset` can output the sample data itself ("value" type), its index ("index" type), or both ("tuple" type), which can then be passed as input to other components or functions[1][3].
*   **Dynamic Updates**: The content of a `Dataset` can be updated dynamically. For instance, a button click could trigger a function that returns a new `gr.Dataset` instance with different samples, updating the UI[1][3].

## Example Usage

Here's how you might display text samples using `gr.Dataset` and update them:

```python
import gradio as gr

initial_samples = [
    ["This is the first example."],
    ["Here is another sample text."]
]

updated_samples = [
    ["Updated sample 1."],
    ["Updated sample 2."]
]

# Function to update the dataset
def update_examples():
    # Return a new Dataset configuration targeting the existing component
    return gr.Dataset(samples=updated_samples)

with gr.Blocks() as demo:
    # Define the component type within the dataset
    textbox_component = gr.Textbox()

    # Initialize the dataset
    example_dataset = gr.Dataset(
        components=[textbox_component],
        samples=initial_samples,
        label="Example Prompts"
    )

    # Button to trigger the update
    update_button = gr.Button("Show Different Examples")

    # Connect button click to update function, targeting the dataset
    update_button.click(fn=update_examples, inputs=None, outputs=[example_dataset])

    # Example of using the selected sample (optional)
    output_textbox = gr.Textbox(label="Selected Example")
    example_dataset.select(fn=lambda x: x[0], inputs=[example_dataset], outputs=[output_textbox])


demo.launch()
```

In this example:
1.  An initial `Dataset` is created with `initial_samples` and a `Textbox` component structure[1][3].
2.  A button is added. When clicked, it calls `update_examples`[1][3].
3.  The `update_examples` function returns a new `gr.Dataset` configuration using `updated_samples`, which Gradio uses to update the `example_dataset` component in the UI[1][3].
4.  The `.select()` event is used so that clicking a sample populates `output_textbox`[3].

The `gr.Dataset` component is distinct from the general Gradio API client (`gradio_client`), which is used for programmatically interacting with a running Gradio application remotely[2][4]. `gr.Dataset` is specifically for displaying example data *within* the UI itself[1][3].

Citations:
[1] https://www.gradio.app/docs/gradio/dataset
[2] https://www.gradio.app/guides/getting-started-with-the-python-client
[3] https://www.gradio.app/4.44.1/docs/gradio/dataset
[4] https://pyimagesearch.com/2025/02/03/introduction-to-gradio-for-building-interactive-applications/
[5] https://www.gradio.app/guides/quickstart
[6] https://www.gradio.app/docs
[7] https://discuss.huggingface.co/t/how-to-use-gradio-api/47108
[8] https://www.youtube.com/watch?v=44vi31hehw4
[9] https://www.reddit.com/r/learnmachinelearning/comments/1co3r40/how_to_use_gradio_via_api_with_curl/
