# Content Sentiment Analyzer

A privacy-first approach to content moderation using Nillion AIVM's secure inference capabilities.

## Overview

Content Sentiment Analyzer is a cutting-edge application that provides real-time sentiment analysis and content moderation while maintaining complete user privacy. By leveraging Nillion AIVM's secure inference capabilities, the application analyzes text content without exposing the actual data, making it ideal for sensitive communications and social media platforms.

### Key Features

- Privacy-preserving sentiment analysis
- Real-time toxicity scoring
- Secure multi-party computation
- End-to-end encryption
- Actionable content improvement suggestions
- Interactive visualization dashboard
- No data storage - complete privacy

## Technical Implementation

### Technology Stack

- Nillion AIVM for secure inference
- Streamlit for the user interface
- BERT-tiny model for sentiment classification
- Plotly for interactive visualizations
- Python for backend processing

### Privacy Features

- End-to-end encryption using Nillion AIVM
- No data storage or logging
- Secure multi-party computation for analysis
- Privacy-preserving model inference

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/content-sentiment-analyzer.git
cd content-sentiment-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the Nillion AIVM devnet:
```bash
aivm-devnet
```

5. Upload the model:
```bash
python upload_model.py
```

6. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Access the web interface at `http://localhost:8501`
2. Enter or paste the content you want to analyze
3. Click "Analyze Content" to receive instant feedback
4. View detailed sentiment analysis, toxicity scoring, and suggestions
5. Use the interactive visualizations to understand the analysis better

## Demo
![App-1](https://github.com/user-attachments/assets/c6cde9a4-2597-44a9-a1b3-750c560be887)

![App-2](https://github.com/user-attachments/assets/e1b8e1fb-68d4-4665-9f55-f2e9576e23df)

![App-3](https://github.com/user-attachments/assets/da21b460-9809-4ed8-8b69-171814870783)
Positive Sentiment Example Above

![App-4](https://github.com/user-attachments/assets/b982bca4-b60e-464b-9b3a-cbf8b6b6c2f9)
Negative Sentiment Example Above

## Future Roadmap

- Social media API integration
- Support for multiple languages
- Enhanced visualization options
- Browser extension development
- Mobile application development

## Acknowledgments

- Nillion Network for providing the AIVM infrastructure
- The open-source community for various tools and libraries used in this project
