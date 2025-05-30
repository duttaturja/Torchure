# Torchure: A Beautiful PyTorch Adventure.


Welcome to **Torchure** – my personal lab notebook as I learn PyTorch by following [!Patrick Loeber](https://github.com/patrickloeber)’s acclaimed tutorial series 📺 ([video](https://www.youtube.com/watch?v=c36lUUr864M)) and the companion repository [`pytorchTutorial`](https://github.com/patrickloeber/pytorchTutorial). Think of this repo as a living, well‑commented scrapbook: each lecture folder contains bite‑sized code experiments, annotated notebooks, and challenges that build up to real‑world deep‑learning projects.

> **Why “Torchure”?**  Because PyTorch + Adventure = *Torchure* (and come on! *do* I really need to explain myself?) 🔥

---

## 🌟 Key Features

| What you’ll find | Why it’s useful |
|-----------------|-----------------|
| **Lecture‑by‑Lecture Modules** | Mirrors the tutorial playlist so you can jump straight to the code that matches the video you’re watching. |
| **Clean, Pedagogical Code** | Short scripts & notebooks focus on *one* concept at a time (tensors, autograd, CNNs, etc.). Each file is heavily commented. |
| **Experiment‑Ready Boilerplate** | Re‑usable training loop, dataset utilities, and config files let you hack quickly without rewiring everything from scratch. |
| **Python = 3.11 / PyTorch ≥ 2.0** | Modern syntax (type hints, pattern matching) and the latest torch goodies (torch.compile, MPS, etc.). |

---

## 📂 Project Structure
```
Torchure/
├── Lecture1/               # Tensor Basics
├── Lecture2/               # Gradient Accumulation             
├── Lecture3/               # Backpropagation
├── Lecture4/               # Gradient Descent
├── Lecture5/               # Training Pipeline      
├── Lecture6/               # Linear Regression
├── Lecture7/               # Logistic Regression
├── Lecture8/               # Dataset & Dataloader         
├── Lecture9/               # Dataset Transform
├── Lecture10/              # Softmax and Cross Entropy
├── Lecture11/              # Activation Fucntion
├── Lecture12/              # Feed Forward Network
├── Lecture13/              # CNN
├── Lecture14/              # Transfer Learning
├── Lecture15/              # Tensorboard
├── Lecture16/              # Save and Load Model
├── main.py                 # Will try to make a Tiny CLI to run any lecture module
├── requirements.txt        # Minimal dependency pin‑file (torch, torchvision, etc.)
├── pyproject.toml          # Optional modern build metadata
├── .python-version         # Local version pin for pyenv / poetry
├── .gitignore
└── README.md               # You are here 🎉
```
> **Heads‑up:** The tree above is trimmed for brevity – dive into each lecture folder for the full source.

---

## 🚀 Quickstart

### 1. Clone & enter the repo
```bash
git clone https://github.com/duttaturja/Torchure.git
cd Torchure
```

### 2. Create an isolated environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
*Alternatively, use **conda** or **pyenv** – anything that gives you Python ≥ 3.10.*

### 3. Install the dependencies
```bash
pip install -r requirements.txt
# or (if you prefer the modern toolchain)
# pip install uv && uv pip install -r requirements.txt
```
This pulls in **PyTorch**, **torchvision**, **tqdm**, **matplotlib**, and other requirements and small helpers.

### 4. Run an example
```bash
python Lecture2/autograd.py   # train the Lecture 2 MLP on your GPU
```
---
| Command                           | Description                                                               |
| --------------------------------- | ------------------------------------------------------------------------- |
| `uv run main.py`                  | Launches the interactive CLI to run any lecture (Lecture01 to Lecture16). |
| Then type `1` to `16`             | Runs the first Python script from the corresponding `LectureXX/` folder.  |
| `exit`                            | Gracefully exits the interactive CLI.                                     |
| `python -m torch.utils.benchmark` | Verifies if PyTorch detects your CPU/GPU correctly.                       |

---

## 📚 Learning Resources
- **Tutorial Playlist:** Patrick Loeber – _"PyTorch – From Zero to Hero"_ → <https://www.youtube.com/watch?v=c36lUUr864M>
- **Original Code:** <https://github.com/patrickloeber/pytorchTutorial>
- **Official Docs:** <https://pytorch.org/docs>
- **Cheat Sheet:** <https://pytorch.org/tutorials/beginner/quickstart_tutorial.html>

---

## 🤝 Contributing
This is primarily a personal study repo, but feel free to open issues for bugs or suggestions – PRs that improve clarity (comments, typos, docs) are welcome!

---

## 📜 License
Licensed under the **MIT License** – see [`LICENSE`](LICENSE) for details.

---

Maintained by **[Turja Dutta](https://github.com/duttaturja/)**

