style_content = """
pre, code {
    background-color: #272822;
}
    .scrollable {
        font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace;
        height: 500px;
        overflow: auto;
    }
    """
dark_mode_gradio_js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
