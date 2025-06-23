# 📊 Financial Document Q&A App (with Gemini + Pinecone)

This Streamlit app allows you to upload **company financial reports (PDFs)** and ask natural language questions about them. You can also generate **automatic visualizations** such as revenue charts across years.

🔗 **Live App**: [https://financial-data-app.streamlit.app](https://financial-data-app.streamlit.app)

---

## 🚀 Features

- 📁 Upload annual reports or any company PDF
- 🔍 Ask questions like:
  - *"What was the net profit in FY 2023?"*
  - *"Plot the revenue trend over the last 3 years"*
- 🧠 Powered by:
  - **LangChain** for chunking and prompt orchestration
  - **Gemini 1.5 Flash** for question answering
  - **Pinecone** for vector storage and semantic search
- 📊 Supports automatic **revenue bar chart** generation

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Google Gemini (via `langchain-google-genai`)](https://ai.google.dev/)
- [Pinecone Vector DB](https://www.pinecone.io/)
- [LangChain](https://www.langchain.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/)

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/financial-data-qa.git
cd financial-data-qa
```

### 2. Set up `.env` file

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_env  # e.g., gcp-starter
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🚀 Deploy on Streamlit Cloud

1. Push your repo to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app → link to GitHub repo
4. In the **Secrets** tab, add:

```env
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_env
```

5. Click **Deploy**

---

## 📚 Example Questions

- "What is the total income for FY 2023?"
- "Plot the revenue trend"
- "Was there an increase in profit between 2022 and 2023?"

---

## 🧑‍💻 Author

Built by [Vivek Gajula](https://github.com/gajula21)

---

## 📝 License

MIT License
