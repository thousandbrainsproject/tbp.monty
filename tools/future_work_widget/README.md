# Future Work Widget

A tool that

1. Processes a JSON file that is produced by the github_readme_sync tool and converts it into a format consumable by the widget.
2. Contains a simple single page app that uses that processed JSON file to construct a dynamic filterable HTML table.

## Setup

First, ensure you've setup Monty. See [Getting Started - 2. Set up Your Environment](https://thousandbrainsproject.readme.io/docs/getting-started#2-set-up-your-environment).

Next, from the root Monty directory, install this tool's dependencies:

```
pip install -e '.[github_readme_sync_tool,future_work_widget_tool]'
```

## Usage

```
python -m tools.future_work_widget.cli --help
usage: cli.py [-h] [--docs-snippets-dir DOCS_SNIPPETS_DIR] index_file output_dir

Build the data and package the future work widget.

positional arguments:
  index_file            The JSON file to validate and transform
  output_dir            The output directory to create and save data.json

optional arguments:
  -h, --help            show this help message and exit
  --docs-snippets-dir DOCS_SNIPPETS_DIR
                        Optional path to a snippets directory for validation files
```


## Tests

```
pytest -n 0 tools/future_work_widget
```

## Running Locally

To try the tool out, simply run the following command from the tbp.monty directory:

```
python tools/future_work_widget/run_local.py
```

And then point your browser to http://localhost:8080

## Widget Configuration

The widget supports URL parameters to customize its display:

### `columns`
Controls which columns are displayed in the table and in what order.

**Usage:** `?columns=column1,column2,column3`

**Available columns:**
- `title` - The title of the future work item with edit/view links
- `estimated-scope` - Size estimation (small, medium, large)
- `improved-metric` - Type of improvement
- `output-type` - Type of output expected
- `rfc` - Related RFC link or reference
- `status` - Current status with contributor avatars
- `tags` - Categorization tags
- `skills` - Required skills for the work

**Examples:**
- Show only title and status: `?columns=title,status`
- Show title, tags, and skills: `?columns=title,tags,skills`
- Show all columns (default): No parameter needed

**Note:** Column names are case-insensitive and whitespace around commas is ignored.

### `q`
Sets the initial search term and enables shareable search URLs.

**Usage:** `?q=search+term`

**Behavior:**
- The search term is automatically populated in the search box when the page loads
- The table is filtered based on the search term on initial load
- As you type in the search box, the URL is updated to reflect the current search
- URLs with search terms can be shared with others

**Examples:**
- Search for "learning": `?q=learning`
- Search for multiple terms: `?q=learning+module`
- Combined with columns: `?columns=title,status&q=performance`

**Note:** The search parameter supports the same multi-word filtering as the search box (all words must be found somewhere in the searchable text).

