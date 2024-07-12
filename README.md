
# install support
https://github.com/openai/whisper/discussions/1463#discussion-5324136
https://hub.tcno.co/ai/whisper/install/

whisper.exe demo.mp4 --model medium --language 'Chinese' --output_format txt --task transcribe


# Whisper

[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://arxiv.org/abs/2212.04356)
[[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.


## Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.


## Setup

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably [OpenAI's tiktoken](https://github.com/openai/tiktoken) for their fast tokenizer implementation. You can download and install (or update to) the latest release of Whisper with the following command:

    pip install -U openai-whisper

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

    pip install git+https://github.com/openai/whisper.git 

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

You may need [`rust`](http://rust-lang.org) installed as well, in case [tiktoken](https://github.com/openai/tiktoken) does not provide a pre-built wheel for your platform. If you see installation errors during the `pip install` command above, please follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install Rust development environment. Additionally, you may need to configure the `PATH` environment variable, e.g. `export PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`, you need to install `setuptools_rust`, e.g. by running:

```bash
pip install setuptools-rust
```


## Available models and languages

There are five model sizes, four with English-only versions, offering speed and accuracy tradeoffs. Below are the names of the available models and their approximate memory requirements and inference speed relative to the large model; actual speed may vary depending on many factors including the available hardware.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |

The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.

Whisper's performance varies widely depending on the language. The figure below shows a performance breakdown of `large-v3` and `large-v2` models by language, using WERs (word error rates) or CER (character error rates, shown in *Italic*) evaluated on the Common Voice 15 and Fleurs datasets. Additional WER/CER metrics corresponding to the other models and datasets can be found in Appendix D.1, D.2, and D.4 of [the paper](https://arxiv.org/abs/2212.04356), as well as the BLEU (Bilingual Evaluation Understudy) scores for translation in Appendix D.3.

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)



## Command-line usage

The following command will transcribe speech in audio files, using the `medium` model:

    whisper audio.flac audio.mp3 audio.wav --model medium

The default setting (which selects the `small` model) works well for transcribing English. To transcribe an audio file containing non-English speech, you can specify the language using the `--language` option:

    whisper japanese.wav --language Japanese

Adding `--task translate` will translate the speech into English:

    whisper japanese.wav --language Japanese --task translate

Run the following to view all available options:

    whisper --help

See [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) for the list of all available languages.


## Python usage

Transcription can also be performed within Python: 

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

Internally, the `transcribe()` method reads the entire file and processes the audio with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window.

Below is an example usage of `whisper.detect_language()` and `whisper.decode()` which provide lower-level access to the model.

```python
import whisper

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
```

## More examples

Please use the [🙌 Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell) category in Discussions for sharing more example usages of Whisper and third-party extensions such as web demos, integrations with other tools, ports for different platforms, etc.


## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.




#offline install discussion: 

Whisper Full (& Offline) Install Process for Windows 10/11

Purpose: These instructions cover the steps not explicitly set out on the main Whisper page, e.g. for those who have never used python code/apps before and do not have the prerequisite software already installed.
Requirements:

    Full admin rights on your computer.
    A PC with a CUDA-capable dedicated GPU with at least 4GB of VRAM (but more VRAM is better). See: Available models and languages
    For online installation: An Internet connection for the initial download and setup.
    For offline installation: Download on another computer and then install manually using the "OPTIONAL/OFFLINE" instructions below.

Installation
Step 1: Unlisted Pre-Requisites

    Before you can run whisper you must download and install the follopwing items. (For offline installation just download the files on another machine and move them to your offline machine to install them.)
        NVIDIA CUDA drivers: https://developer.nvidia.com/cuda-downloads
        Python 3.9 or 3.10 (x64 version) from https://www.python.org/ (Whisper claims to run with >3.7 but as of 2023-01-18 some dependencies require >3.7 but <3.11).
        FFMPEG
            To install via Scoop (https://scoop.sh/), in powershell run
                Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
                irm get.scoop.sh | iex
                scoop install ffmpeg
            OPTIONAL/OFFLINE: Follow instructions here: How to install and use FFMPEG and make sure not to skip the part about adding FFMPEG to the Windows PATH variable.
        Git for windows from https://gitforwindows.org/
    Reboot after installing these items.

Step 2B: Whipser Install (Online Install for Online Use)

    Open a command prompt and type this command:
        pip install git+https://github.com/openai/whisper.git
    You may now use Whisper online and no further steps are required.

Step 2B: Whipser Install (Online Install for later Offline Use)

    Open a command prompt and type these commands:
        pip install git+https://github.com/openai/whisper.git
        pip install blobfile
    Continue to Step 3: Download Other Required Files

Step 2C: Whipser Install (Offline Install for later Offline Use)

    Option 1: Get the most up to date version of Whisper:
        Install Python and Git from Step 1 on an second computer you can connect to the internet and reboot to ensure both are working.
        On the ONLINE machine open a command prompt in any empty folder and type the following commands:
            pip download git+https://github.com/openai/whisper.git
            pip download blobfile
    Option 2: Download all the necessary files from here OPENAI-Whisper-20230314 Offline Install Package
    Copy the files to your OFFLINE machine and open a command prompt in that folder where you put the files, and run
        pip install openai-whisper-20230314.zip (note the date may have changed if you used Option 1 above).
        pip install blobfile-2.0.2-py3-none-any.whl. (note the version may have changed if you used Option 1 above).
    Continue to Step 3: Download Other Required Files

Step 3: Download Other Required Files (for Offline Use)

    Download Whisper's Language Model files place them in C:\Users[Username]\.cache\whisper Note: If the links are dead updated links can be found at lines 17-27 here: init.py
        Tiny.En
        Tiny
        Base.En
        Base
        Small.En
        Small
        Medium.En
        Medium
        Large-v1
        Large-v2 (Annoucing the large-v2 model)
    Download Whisper's vocabulary and encoder files. (Per issue 1399).
        Download Vocab.bpe
        Download Encoder.json
        Install the files to a folder of your choosing, e.g. C:\Users[Username]\.cache\whisper.
        Update file links in your local copy of openai_public.py which will be installed in your python folder e.g. C:\Users[UserName]\AppData\Local\Programs\Python\Python310-32\Lib\site-packagespython3.9/site-packages/tiktoken_ext/openai_public.py to point to where you downloaded the files.
            Remove the URL "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/" and replace it with your local copy, e.g. "C:/Users/[Username]/.cache/whisper/vocab.bpe" and "C:/Users/[Username]/.cache/whisper/encoder.json"

def gpt2():
    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file="C:/Users/nic/.cache/whisper/vocab.bpe",
        encoder_json_file="C:/Users/nic/.cache/whisper/encoder.json",
    )

Alternative Offline Method

See the pre-compiled .exe version of Whisper provided here: Purfview / Whisper Standalone

