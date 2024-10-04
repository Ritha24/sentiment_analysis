import streamlit as st
import pandas as pd
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from openai import OpenAI
import concurrent.futures
import re
from pandas import json_normalize
import plotly.graph_objects as go
import plotly.express as px


logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key='your-api-key')

class TranscriptAnalysis:
    def __init__(self, client):
        self.client = client
        self.transcription_text = ''
    
    def LLMClient(self, prompt):
        try:
            chat_completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"API request failed: {e}")
            return ''
    
    def analyze_transcription(self):
        prompt = f"""
        Analyze the conversation with high sensitivity to emotional content. Return a detailed sentiment score and list out the topics and top keywords of the conversation in the JSON format provided below. Do not return anything else.
        CRITICAL: Be very conservative with neutral scoring. Even slight emotional indicators should shift the score away from neutral.
        Generate 6 main keywords related to the conversation.
        Analyze and provide the JSON in this format:
        {{
            "summary": "Rewrite the below audio transcript into less than 5000 characters. You can remove stop words or other irrelevant words to achieve this. Retain the phone conversation format. This is needed for training the BIRT model.",
            "sentiment": "A descriptive word that best captures the overall sentiment of the conversation (e.g., happy, sad, angry, excited, etc.). Don't return positive, negative, or neutral.",
            "neg": 0.0,
            "neu": 0.0,
            "pos": 0.0,
            "compound": 0.0,
            "topics": ["topic1", "topic2", "topic3", "topic4"],
            "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"],
            "category": "Categorize what the transcription is about (e.g., Customer support, Sales, Marketing, IT support, etc.)",
            "strength_areas": ["List the areas or aspects of the conversation that were handled well or positively. Even if sentiment is neutral, provide aspects where the conversation was clear, professional, or effective in a single word."],
            "weak_areas": ["List the areas or aspects of the conversation that need improvement. Even if sentiment is neutral, provide areas where the conversation could be enhanced in a single word (e.g., tone, clarity, structure, or content)."],
        }}
        Make sure you follow the below guidelines:
        1. The sum of "neg", "neu", and "pos" must equal 1.0.
        2. Limit "neu" to a maximum of 0.3, even for seemingly neutral content.
        3. For any emotional indicators (positive or negative), ensure the corresponding score is at least 0.4.
        4. The "compound" score should be very sensitive to sentiment, using the full range from -1 to 1.
        5. In absence of clear sentiment, lean towards the slightest detected emotion rather than neutral.
        6. For summary, rewrite the audio transcript into less than 5000 characters. You can remove stop words or other irrelevant words to achieve this. Retain the phone conversation format. This is needed for training the BIRT model.
        7. If sentiment is neutral, still identify strengths and areas for improvement based on clarity, tone, and professionalism.
        Here is the conversation: '{self.transcription_text}'
        """
        return self.LLMClient(prompt)

def is_valid_transcript(transcript):
    if len(transcript) < 600:
        return False
    return True

transcript_analysis = TranscriptAnalysis(client)
def process_transcript(row):
    if not is_valid_transcript(row['TRANSCRIPT']):
        return None

    transcript_analysis.transcription_text = row['TRANSCRIPT']
    sentiment_result = transcript_analysis.analyze_transcription()
    try:
        sentiment_json = json.loads(sentiment_result)
        return {
            'TRANSCRIPT': row['TRANSCRIPT'],
            'sentiment_analysis': sentiment_result,
            'sentiment_neg': sentiment_json.get('neg', 0.0),
            'sentiment_neu': sentiment_json.get('neu', 0.0),
            'sentiment_pos': sentiment_json.get('pos', 0.0),
            'sentiment_compound': sentiment_json.get('compound', 0.0),
            'topics': sentiment_json.get('topics', []),
            'keywords': sentiment_json.get('keywords', []),
            'transcription_category': sentiment_json.get('category', ''),
            'sentiment': sentiment_json.get('sentiment', ''),
            'sentiment_category': 'Positive' if sentiment_json.get('compound', 0.0) > 0 else ('Negative' if sentiment_json.get('compound', 0.0) < 0 else 'Neutral'),
            'transcription_summary': sentiment_json.get('summary', '')
        }
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing sentiment result: {e}")
        return None

def condense_strengths_weaknesses(counter, threshold=5):
    """Condense the strength and weakness areas based on a frequency threshold."""
    condensed_summary = {}
    for key, count in counter.items():
        if count >= threshold:
            condensed_summary[key] = count
        else:
            if "Other" not in condensed_summary:
                condensed_summary["Other"] = 0
            condensed_summary["Other"] += count
    return condensed_summary

def summarize_sentiments(data):
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    topics_counter = Counter()
    category_counter = Counter()
    keywords_counter = Counter()
    strength_areas_counter = Counter()
    weak_areas_counter = Counter() 

    for index, row in data.iterrows():
        try:
            sentiment_json = json.loads(row['sentiment_analysis'])
            sentiment_compound = sentiment_json.get('compound', 0.0)
            
            if sentiment_compound > 0:
                positive_count += 1
            elif sentiment_compound < 0:
                negative_count += 1
            else:
                neutral_count += 1
            
            topics_counter.update(sentiment_json.get('topics', []))
            keywords_counter.update(sentiment_json.get('keywords', []))
            strength_areas_counter.update(sentiment_json.get('strength_areas', []))
            weak_areas_counter.update(sentiment_json.get('weak_areas', []))
            category = sentiment_json.get('category')
            if category:
                category_counter[category] += 1 
                 
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON in row {index}: {e}")
            continue

    total_count = positive_count + neutral_count + negative_count
    sentiment_distribution = {
        'positive': (positive_count / total_count) * 100 if total_count > 0 else 0,
        'neutral': (neutral_count / total_count) * 100 if total_count > 0 else 0,
        'negative': (negative_count / total_count) * 100 if total_count > 0 else 0
    }

    condensed_strengths = condense_strengths_weaknesses(strength_areas_counter, threshold=5)
    condensed_weaknesses = condense_strengths_weaknesses(weak_areas_counter, threshold=5)

    summary = {
        'overall_sentiment': 'positive' if positive_count > negative_count else 'negative' if negative_count > positive_count else 'neutral',
        'sentiment_distribution': sentiment_distribution,
        'topics': dict(topics_counter),
        'category': dict(category_counter),
        'keywords': dict(keywords_counter),
        'strength_areas': condensed_strengths,
        'weak_areas': condensed_weaknesses 

    }

    return summary

def summarize_with_openai(summary):
    analyzer = TranscriptAnalysis(client)
    prompt = f"""
    Analyze the following set of conversation transcriptions and provide a comprehensive summarized analysis. 
    Focus on overall sentiment trends, topics, keywords, and categories. Use all available information to provide 
    insights that will be useful for the company to improve their services.

    Overall Sentiment: {summary['overall_sentiment']}
    Sentiment Distribution: {summary['sentiment_distribution']}

    Top keywords and their occurrences:
    {dict(Counter(summary['keywords']).most_common(20))}

    Categories and their occurrences:
    {dict(Counter(summary['category']).most_common(10))}

    Top strength areas: {summary['strength_areas'] }

    Top weak areas: {summary['weak_areas']}

    Return your analysis in JSON format with the following structure:

    {{
        "overall_sentiment": "string describing the general sentiment across all transcriptions",
        "sentiment_distribution": {{
            "positive": float (percentage of positive sentiments),
            "neutral": float (percentage of neutral sentiments),
            "negative": float (percentage of negative sentiments)
        }},
        "top_categories_and_keywords": [
            {{
                "category": "string (name of the category)",
                "percentage": float (percentage of transcripts in this category),
                "top_keywords": [
                    {{
                        "keyword": "string (specific, relevant keyword for the category)",
                        "occurrences": int (number of occurrences of this keyword within the category),
                        "context": "string (brief example or context of keyword usage)"
                    }},
                    ...
                ]
            }},
            ...
        ],
        "top_topics": [
            {{
                "topic": "string (specific topic)",
                "relevance_score": float (0-1 indicating importance of topic),
                "related_keywords": ["string", "string", ...],
            }},
            ...
        ],
        "top_strength_areas": [
            {{
                "area": "string (specific strength area and don't display other)",
                "frequency": int (number of occurrences),
                "impact": "string (brief description of the positive impact)"   
            }},
            ...
        ],
        "top_weak_areas": [
            {{
                "area": "string (specific weak area and don't display None)",
                "frequency": int (number of occurrences),
                "improvement_suggestion": "string (brief suggestion for improvement)"
            }},
            ...
        ],
        "insights_for_improvement": [
            {{
                "insight": "string describing an insight or recommendation based on the analysis",
                "priority": "string (high, medium, low)",
                "impact_area": "string (e.g., customer satisfaction, operational efficiency)",
                "supporting_data": "string (brief data points supporting this insight)"
            }},
            ...
        ]
    }}

    Guidelines for generating accurate keywords and topics:
    1. Ensure keywords are specific and relevant to each category. Avoid generic terms.
    2. Consider multi-word phrases as keywords when they represent a single concept.
    3. Focus on actionable and insightful keywords that provide clear value for improvement.
    4. Differentiate between similar keywords by considering their context and sentiment.
    5. Identify emerging trends or unusual patterns in the keywords.
    6. For each keyword, provide a brief context or example of its usage to ensure accuracy.
    7. Assign a sentiment (positive, neutral, negative) to each keyword based on its typical usage.
    8. Calculate a relevance score for topics based on frequency and importance in the conversations.

    Ensure that your response is a valid JSON object. For the 'top_categories_and_keywords' field, provide the top 10 categories, and for each category, list the top 10 keywords within that category. For 'top_topics', list the top 15 most relevant topics across all categories. For 'top_strength_areas' and 'top_weak_areas', provide at least 10 items each. Ensure that the keywords listed under each category are strictly relevant to that specific category and provide valuable insights for improvement.
    """

    return analyzer.LLMClient(prompt)

def create_improved_nested_pie_chart(data):
    categories = []
    keywords = []
    category_sizes = []
    keyword_sizes = []
    
    for category in data['top_categories_and_keywords']:
        categories.append(category['category'])
        category_sizes.append(category['percentage'])
        
        for keyword in category['top_keywords']:
            keywords.append(f"{keyword['keyword']} ({keyword['occurrences']})")
            keyword_sizes.append(keyword['occurrences'])
    
    fig, ax = plt.subplots(figsize=(20, 18))
    
    category_colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    outer_colors = plt.cm.Pastel1(np.linspace(0, 1, len(keyword_sizes)))
    
    wedges, texts = ax.pie(keyword_sizes, labels=keywords, colors=outer_colors, radius=1,
                           wedgeprops=dict(width=0.3, edgecolor='white'),
                           labeldistance=1.05)
    
    for i, text in enumerate(texts):
        angle = (wedges[i].theta2 + wedges[i].theta1) / 2
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        text.set_rotation_mode("anchor")
        text.set_rotation(angle)
        text.set_ha(horizontalalignment)
        text.set_va("center")
        
        arrow_length = 0.2
        label_distance = 1.2
        
        ax.annotate(text.get_text(), xy=(x, y), xytext=(label_distance*np.sign(x), label_distance*y),
                    horizontalalignment=horizontalalignment,
                    verticalalignment="center",
                    size=8, arrowprops=dict(arrowstyle="-",
                                            connectionstyle=f"angle,angleA=0,angleB={angle}",
                                            color="gray",
                                            shrinkA=3,
                                            shrinkB=3))
    
    for text in texts:
        text.set_visible(False)
    
    inner_wedges, inner_texts, inner_autotexts = ax.pie(category_sizes, labels=categories, colors=category_colors, radius=0.7,
                                                        wedgeprops=dict(width=0.4, edgecolor='white'),
                                                        labeldistance=0.6, autopct='%1.1f%%', pctdistance=0.75)
    
    for text in inner_texts:
        text.set_fontweight('bold')
        text.set_size(10)
    
    center_circle = plt.Circle((0, 0), 0.3, fc='white')
    ax.add_artist(center_circle)
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=category_colors[i], edgecolor='none') for i in range(len(categories))]
    ax.legend(legend_elements, categories, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3, title="Categories")
    
    plt.title("Categories and Keywords Distribution", fontsize=20, pad=20)
    plt.axis('equal')
    plt.tight_layout()
    return fig

def display_categories_and_keywords(data):
    categories_keywords = []
    for category in data["top_categories_and_keywords"]:
        category_percentage = category['percentage']
        for keyword in category["top_keywords"]:
            categories_keywords.append({
                "Category": category["category"],
                "Category %": f"{category_percentage:.2f}%",
                "Keyword": keyword["keyword"],
                "Occurrences": keyword["occurrences"],
                "Context": keyword["context"]
            })
    
    categories_keywords_df = pd.DataFrame(categories_keywords)
    
    # st.subheader("Top Categories and Keywords")
    
    # Apply custom styling
    st.markdown("""
    <style>
    .dataframe {
        font-size: 12px;
    }
    .dataframe th {
        background-color: #4a4a4a;
        color: white;
        font-weight: bold;
        text-align: left;
    }
    .dataframe td {
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    st.dataframe(categories_keywords_df)

def save_json_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info(f"JSON data saved to {filename}")

def load_json_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    logging.info(f"JSON data loaded from {filename}")
    return data      

def main():
    if "get_analysis" not in st.session_state:
        st.session_state["get_analysis"] = False
    if "get_charts" not in st.session_state:
        st.session_state["get_charts"] = False

    st.set_page_config(page_title="Transcript Analysis Dashboard", layout="wide")
    st.title("Transcript Analysis Dashboard")

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    run_all = st.button('Analyze Transcripts')
    if run_all:
        st.session_state["get_analysis"] = True
        st.session_state["get_charts"] = True

    tab1, tab2 = st.tabs(["Analysis", "Charts"])

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        
        if st.session_state["get_analysis"] or st.session_state["get_charts"]:
            with st.spinner("Analyzing transcripts..."):
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    results = list(executor.map(process_transcript, [row for index, row in data.iterrows()]))
                
                results = [result for result in results if result is not None]
                results_df = pd.DataFrame(results)
                
                summary_json = summarize_sentiments(results_df)
                openai_summary = summarize_with_openai(summary_json)
                openai_data = json.loads(openai_summary)
                save_json_to_file(json.loads(openai_summary), 'openai_summary.json')
               
                st.success("Analysis complete!")

        if st.session_state["get_analysis"]:
            with tab1:
                st.header("Overall Sentiment Analysis")
                sentiment_df = pd.DataFrame([openai_data['sentiment_distribution']])
                st.dataframe(sentiment_df)
                
                st.header("Top Categories and Keywords")
                display_categories_and_keywords(openai_data)
                
                st.header("Top Topics")
                topics_df = json_normalize(openai_data['top_topics'])
                st.dataframe(topics_df)
                
                st.header("Strengths and Weaknesses")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top Strength Areas")
                    strengths_df = json_normalize(openai_data['top_strength_areas'])
                    st.dataframe(strengths_df)
                with col2:
                    st.subheader("Top Weak Areas")
                    weaknesses_df = json_normalize(openai_data['top_weak_areas'])
                    st.dataframe(weaknesses_df)
                
                st.header("Insights for Improvement")
                insights_df = json_normalize(openai_data['insights_for_improvement'])
                st.dataframe(insights_df)

                st.session_state["get_analysis"] = False

        if st.session_state["get_charts"]:
            with tab2:
                st.header("Sentiment Distribution")
                fig = px.pie(sentiment_df.T, values=0, names=sentiment_df.columns, title="Sentiment Distribution")
                st.plotly_chart(fig)

                st.header("Top Topics")
                fig = px.bar(topics_df, x='topic', y='relevance_score', title="Top Topics by Relevance Score")
                st.plotly_chart(fig)

                st.header("Strengths and Weaknesses")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(strengths_df, x='area', y='frequency', title="Top Strength Areas")
                    st.plotly_chart(fig)
                with col2:
                    fig = px.bar(weaknesses_df, x='area', y='frequency', title="Top Weak Areas")
                    st.plotly_chart(fig)

                st.header("Insights for Improvement")
                priority_map = {"low": 1, "medium": 2, "high": 3}
                insights_df['priority_num'] = insights_df['priority'].map(priority_map)
                fig = px.scatter(insights_df, 
                                 x='priority', 
                                 y='impact_area', 
                                 size='priority_num',
                                 color='priority',
                                 hover_name='insight', 
                                 title="Insights by Priority and Impact Area",
                                 size_max=20)
                fig.update_layout(
                    xaxis_title="Priority",
                    yaxis_title="Impact Area",
                    legend_title="Priority Level"
                )
                st.plotly_chart(fig)

                st.header("Categories and Keywords Distribution")
                fig = create_improved_nested_pie_chart(openai_data)
                st.pyplot(fig)

                st.session_state["get_charts"] = False
    else:
        st.write("No file uploaded")

if __name__ == "__main__":
    main()