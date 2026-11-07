import io

import pytest

from word2img.sign import main
from word2img.signs import SignLanguageResult


@pytest.fixture
def gif_result():
    Image = pytest.importorskip('PIL.Image')
    output = io.BytesIO()
    with Image.new('RGB', (16, 16), 'red') as first, Image.new('RGB', (16, 16), 'blue') as second:
        first.save(output, format='GIF', save_all=True, append_images=[second],
                   duration=[100, 200], loop=0)
    return SignLanguageResult(image_bytes=output.getvalue())


def test_cli_default_animated_png(tmp_path, monkeypatch, gif_result):
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    def generate(sentence):
        assert sentence == 'Hei, hvordan går det?'
        return gif_result
    monkeypatch.setattr('word2img.sign.sign_language', generate)
    assert main(['Hei, hvordan går det?']) == 0
    with Image.open(next(tmp_path.glob('*.png'))) as image:
        assert image.format == 'PNG'
        assert image.n_frames == 2
        assert image.info['loop'] == 0
        assert image.info['duration'] == 100
        image.seek(1)
        assert image.info['duration'] == 200
        assert image.convert('RGB').getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize('suffix', ['png', 'gif'])
def test_explicit_output(tmp_path, monkeypatch, gif_result, suffix):
    monkeypatch.setattr('word2img.sign.sign_language', lambda _: gif_result)
    output = tmp_path / f'chosen name.{suffix}'
    assert main(['hei', '-o', str(output)]) == 0
    expected = b'\x89PNG\r\n\x1a\n' if suffix == 'png' else b'GIF89a'
    assert output.read_bytes().startswith(expected)


def test_generation_error(monkeypatch, capsys):
    def fail(_):
        raise ValueError('Cannot sign this word')
    monkeypatch.setattr('word2img.sign.sign_language', fail)
    assert main(['unknown']) == 1
    assert 'Cannot sign this word' in capsys.readouterr().err


def test_unsupported_extension():
    with pytest.raises(SystemExit) as exc:
        main(['hei', '-o', 'output.jpg'])
    assert exc.value.code == 2
