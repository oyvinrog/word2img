import io

import pytest

from word2img import sign_language
from word2img import signs


def test_phrase_forms_compounds_and_fingerspelling():
    index = {key: (key, key) for key in ['god morgen', 'gå', 'barn', 'skole', 'æ', 'ø', 'å']}
    result = signs._resolve('God morgen! gikk barneskole æøå', index, True)
    assert [item['sign'] for item in result] == ['god morgen', 'gå', 'barn', 'skole', 'æ', 'ø', 'å']
    assert [item['fingerspelled'] for item in result] == [False] * 4 + [True] * 3
    with pytest.raises(ValueError, match='gikk'):
        signs._resolve('gikk', index, False)
    with pytest.raises(ValueError, match='xyz'):
        signs._resolve('god morgen xyz', index, True)


def test_dictionary_aliases_prefer_exact_and_unqualified(tmp_path):
    path = tmp_path / 'dictionary.xml'
    path.write_text('''<root>
      <leksem visningsord="a (to hender)" filnavn="two"/>
      <leksem visningsord="a 1" filnavn="one"/>
      <leksem visningsord="hei 1" filnavn="variant"/>
      <leksem visningsord="hei" filnavn="exact"/>
      <leksem visningsord="adferd/atferd" filnavn="alias"/>
      <leksem visningsord="empty"/>
    </root>''', encoding='utf-8')
    index = signs._load_index(path)
    assert index['a'][1] == 'one'
    assert index['hei'][1] == 'exact'
    assert index['atferd'][1] == 'alias'
    assert 'empty' not in index


@pytest.mark.parametrize('value', ['', [], [' '], '!?'])
def test_empty_input(value):
    with pytest.raises(ValueError):
        sign_language(value)


@pytest.mark.parametrize('kwargs', [{'fps': 0}, {'fps': True}, {'size': (0, 10)}, {'max_frames': 0}])
def test_bad_options(kwargs):
    with pytest.raises(ValueError):
        sign_language('hei', **kwargs)


def test_api_caches_downloads_and_preserves_order(tmp_path, monkeypatch):
    pytest.importorskip('PIL')
    pytest.importorskip('imageio_ffmpeg')
    calls = []
    def fetch(url, path):
        calls.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'<root><leksem visningsord="hei" filnavn="a b"/></root>'
                         if url == signs.DICTIONARY_URL else b'video')
    monkeypatch.setattr(signs, '_fetch', fetch)
    def encode(clips, size, fps, max_frames):
        assert len(clips) == 2 and clips[0] == clips[1]
        return b'GIF89a'
    monkeypatch.setattr(signs, '_encode_gif', encode)
    result = sign_language(['hei', 'hei'], cache_dir=tmp_path)
    assert result['mime_type'] == 'image/gif'
    assert result['text'] == 'hei hei'
    assert len(result['signs']) == 2
    assert calls == [signs.DICTIONARY_URL, signs.VIDEO_BASE_URL + 'a%20b.mp4']
    sign_language('hei hei', cache_dir=tmp_path)
    assert len(calls) == 2
    sign_language('hei hei', cache_dir=tmp_path, refresh=True)
    assert len(calls) == 3


def test_actual_video_to_gif(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    ffmpeg = pytest.importorskip('imageio_ffmpeg')
    clip = tmp_path / "video with ' spaces.mp4"
    writer = ffmpeg.write_frames(str(clip), (32, 32), fps=10)
    writer.send(None)
    try:
        for color in [(255, 0, 0), (0, 0, 255)]:
            for _ in range(5):
                writer.send(bytes(color) * (32 * 32))
    finally:
        writer.close()
    data = signs._encode_gif([clip, clip], (64, 48), 10, 100)
    with Image.open(io.BytesIO(data)) as gif:
        assert gif.format == 'GIF'
        assert gif.size == (64, 48)
        assert gif.info['loop'] == 0
        duration = 0
        colors = []
        for i in range(gif.n_frames):
            gif.seek(i)
            duration += gif.info['duration']
            colors.append(gif.convert('RGB').getpixel((32, 24)))
        assert duration == 2000
        assert colors[0][0] > 200 and colors[-1][2] > 200
    with pytest.raises(ValueError, match='max_frames'):
        signs._encode_gif([clip], (32, 32), 10, 2)
