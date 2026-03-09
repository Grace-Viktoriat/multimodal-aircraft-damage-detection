Multimodal Aircraft Damage Detection and Captioning
 Project Overview
This project implements a multimodal deep learning system designed for automated structural health monitoring (SHM) in aviation. By fusing computer vision and natural language processing, the system moves beyond simple detection into context-aware damage interpretation.
The system performs three critical tasks:
1.  Classification: Categorizing structural damage (e.g., Dents vs. Cracks).
2.  Captioning: Generating automated natural language descriptions of the visual state.
3.  Summarization: Producing concise damage reports for maintenance workflows.

Problem Statement
Manual aircraft inspection is labor-intensive and prone to subjectivity. This research explores how Transfer Learning and Vision-Language Models (VLMs) can assist inspectors by providing:
Automated Defect Classification: Reducing initial triage time.
AI-Assisted Reporting: Streamlining the documentation of findings for maintenance crews.

System Architecture
- Damage Classification Pipeline
Base Model: VGG16 (Pre-trained on ImageNet).
Strategy: Transfer Learning (Frozen feature extraction and Custom Head).
Architecture Details:
    Dual Dense layers (512 units) with Dropout (0.3) for regularization.
    Optimization:Adam ($lr = 1e-4$) with Binary Crossentropy loss.
Framework: TensorFlow / Keras.

- Vision-Language Generation
Model: BLIP (Bootstrapping Language-Image Pretraining).
Framework: Hugging Face Transformers (PyTorch-backed).
Integration: Implementation of a custom Keras wrapper for BLIP inference, demonstrating cross-framework interoperability between TensorFlow and PyTorch.

Performance & Results
 - Baseline Metrics
      Classification Test Accuracy:  ~68.75%
      Dataset:  Aircraft structural damage (dent/crack classes).
      Input Resolution:  224x224 RGB.
 - Multimodal Inference Samples
      Caption Output:  "this is a picture of a plane that was sitting on the ground in a field"
      Summary Output:  "this is a detailed photo showing the damage to the fuselage of the aircraft"

Note: Performance reflects a baseline model. Future iterations will focus on fine-tuning the VGG16 convolutional blocks to capture higher-resolution edge features typical of hairline cracks.

Tech Stack
Languages: Python
Deep Learning: TensorFlow, Keras, PyTorch
Models: VGG16, BLIP Transformer
Libraries: Hugging Face Transformers, NumPy, Matplotlib, PIL


Quick Start
1.  Clone the Repository:
    ```bash
    git clone [https://github.com/Grace-Viktoriat/multimodal-aircraft-damage-detection.git](https://github.com/ Grace-Viktoriat/multimodal-aircraft-damage-detection.git)
    ```
2.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run Inference: Open `aircraft_damage_classification_and_captioning.ipynb` to view the full pipeline.

Project Structure
multimodal-aircraft-damage-detection/
│
├── aircraft_damage_classification_and_captioning.ipynb
├── README.md
└── requirements.txt

Key Takeaways
This project demonstrates:
•	Transfer learning for practical industrial classification
•	Integration of vision and language models
•	Custom Keras layer development
•	Multimodal AI system design
•	Real-world AI application in aviation safety

Future Roadmap
Model Optimization: Fine-tuning transformer attention heads for aviation-specific vocabulary.
Object Detection: Integrating YOLOv10 for localized bounding boxes.
Deployment: Building a Fast API-based web interface for real-time image uploads. 

Author
Julie Mireille Bokonden
Deep Learning | Multimodal AI | Applied Research
