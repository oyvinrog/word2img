# word2img

Generate an image from a list of words using OpenAI's `gpt-image-1` model.

## Requirements

- Python 3.10+

## Install

```bash
pip install -e .
```

## API usage

```python
from word2img import words_to_img

result = words_to_img(["frog", "does", "dancing"], api_key="your_api_key")
with open("frog-does-dancing.png", "wb") as f:
    f.write(result["image_bytes"])
```

The prompt sent to the model is the hyphen-joined string: `frog-does-dancing`.

## CLI usage

```bash
python -m word2img
```

You will be prompted for:
- Comma-separated words (for example `frog,does,dancing`)
- OpenAI API key on first run only (hidden input)

The CLI stores the API key in the system keyring and writes `<prompt>.png` to the current directory.

## EFF passphrase + image mnemonic

Generate an EFF-style passphrase using computer randomness, then generate an image for it:

```bash
python -m word2img.effgen
```

Or via installed script:

```bash
word2img-effgen
```

Options:
- `-n`, `--num-words`: number of words (default: `6`)

On first run, the tool downloads and caches the official EFF large wordlist at `~/.cache/word2img/eff_large_wordlist.txt`.
