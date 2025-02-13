# Meeting Minutes Creator

## Description

An AI-powered tool that converts audio recordings into structured meeting minutes. It utilizes OpenAI's Whisper for transcription and Meta's LLaMA for text processing.

## Overview

This project is an AI-powered meeting minutes generator. It uses OpenAI's Whisper model for speech-to-text transcription and Meta's LLaMA model to generate structured meeting minutes in Markdown format.

## Features

- Converts audio recordings to text using Whisper ASR.
- Generates well-structured meeting minutes with summaries, key points, and action items.
- Uses LLaMA for text processing and formatting.
- Supports 4-bit quantization for efficient model inference.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- PyTorch with CUDA (for GPU acceleration, if available)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- `bitsandbytes` for 4-bit quantization (Linux required)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/meeting-minutes-creator.git
   cd meeting-minutes-creator
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

1. Authenticate with Hugging Face:
   ```bash
   export HF_TOKEN=your_huggingface_api_key
   ```
2. Run the script:
   ```bash
   python meeting_minutes_creator.py
   ```
3. Follow on-screen prompts to process an audio file.

## Troubleshooting

- If you encounter `bitsandbytes` import errors, ensure you are on a **Linux** system and install the latest version:
  ```bash
  pip install -U bitsandbytes
  ```
- For macOS, 4-bit quantization may not work. Consider using CPU inference instead.

## Contributing

Feel free to submit issues or pull requests to improve the project.

