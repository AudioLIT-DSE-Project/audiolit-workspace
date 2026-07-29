# audiolit-workspace

Core workspace repository for the AudioLIT project.

## Core Service Structure

- `/services` - Contains backend business logic.
- `/gateway` - API Gateway handling client routing.
- `/config` - Global application configurations.
- `/shared` - Common utilities shared across services.

## Audio Upload Endpoint

The backend exposes `POST /api/audio/upload` for streaming binary audio uploads.

Example `curl` command:

```bash
curl -X POST "http://localhost:8000/api/audio/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.wav;type=audio/wav"
```

Postman raw body example:

- Choose `POST`
- URL: `http://localhost:8000/api/audio/upload`
- Body: `form-data`
- Add a field named `file`
- Select `File`
- Choose a `.wav` or `.mp3`
- Set type to `audio/wav` or `audio/mpeg`

Successful response schema:

```json
{
  "transaction_token": "aud_<uuid>",
  "file_path": "/tmp/audiolit/uploads/aud_<uuid>_xxxxx.tmp",
  "audio_config": {
    "format": "wav",
    "sample_rate_hz": 48000,
    "channels": 2,
    "duration_sec": 12.345,
    "total_bytes": 1234567
  },
  "status": "uploaded"
}
```

## Verification

Start the service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -i http://localhost:8000/api/healthz
```

Audio upload:

```bash
curl -i -X POST "http://localhost:8000/api/audio/upload" \
  -F "file=@test.wav;type=audio/wav"
```

Invalid upload validation check:

```bash
curl -i -X POST "http://localhost:8000/api/audio/upload" \
  -F "file=@fake.txt;type=text/plain"
```

CORS preflight example:

```bash
curl -i -X OPTIONS "http://localhost:8000/api/audio/upload" \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: POST"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
