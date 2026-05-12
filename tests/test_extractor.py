import base64
from unittest.mock import patch, MagicMock
from app.vision.extractor import extract_from_bytes


def _fake_completion(content: str):
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


@patch("app.vision.extractor.client")
def test_extract_from_bytes_returns_content(mock_client):
    mock_client.chat.completions.create.return_value = _fake_completion("Invoice total: $500")

    result = extract_from_bytes(b"fake_image_bytes", mime="image/png")

    assert result == "Invoice total: $500"
    mock_client.chat.completions.create.assert_called_once()


@patch("app.vision.extractor.client")
def test_extract_strips_whitespace(mock_client):
    mock_client.chat.completions.create.return_value = _fake_completion("  some text  \n")

    result = extract_from_bytes(b"bytes", mime="image/jpeg")

    assert result == "some text"


@patch("app.vision.extractor.client")
def test_extract_sends_correct_mime(mock_client):
    mock_client.chat.completions.create.return_value = _fake_completion("ok")

    extract_from_bytes(b"bytes", mime="image/jpeg")

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    image_content = messages[0]["content"][1]
    assert "image/jpeg" in image_content["image_url"]["url"]
