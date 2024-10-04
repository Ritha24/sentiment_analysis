import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import json
import logging
from collections import Counter
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
import concurrent.futures

logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
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

def process_transcript(row):
    transcript_analysis = TranscriptAnalysis(client)
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

def create_sentiment_chart(sentiment_dist):
    colors = ['#636EFA', '#00CC96', '#EF553B', '#AB63FA', '#FFA15A']
    
    fig = go.Figure(data=[go.Pie(
        labels=list(sentiment_dist.keys()),
        values=list(sentiment_dist.values()),
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        hoverinfo='label+percent',
        textinfo='value+percent',
        textfont_size=14,
        domain=dict(x=[0, 0.5])  # Adjust the domain to move the pie chart to the left
    )])
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig

def create_category_chart(categories_data):
    categories = []
    percentages = []
    
    if isinstance(categories_data, dict):
        # If it's a dictionary, convert it to the expected format
        categories_data = [{"category": k, "percentage": v} for k, v in categories_data.items()]
    
    for category in categories_data:
        if isinstance(category, dict):
            categories.append(category.get('category', ''))
            percentages.append(category.get('percentage', 0))
        else:
            # If it's not a dictionary, assume it's already in the correct format
            categories.append(category)
            percentages.append(categories_data[category])
    
    df = pd.DataFrame({'Category': categories, 'Percentage': percentages})
    fig = px.bar(df, x='Category', y='Percentage', title="Top Categories")
    return fig

def create_topics_chart(topics):
    df = pd.DataFrame(topics)
    fig = px.scatter(df, x='topic', y='relevance_score', size='relevance_score', 
                     hover_data=['related_keywords'], title="Topic Relevance")
    return fig

def load_json_from_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    logging.info(f"JSON data loaded from {filename}")
    return data

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
   
    fig, ax = plt.subplots(figsize=(15, 13))
   
    category_colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    outer_colors = plt.cm.Pastel1(np.linspace(0, 1, len(keyword_sizes)))
   
    ax.pie(keyword_sizes, labels=keywords, colors=outer_colors, radius=1,
           wedgeprops=dict(width=0.3, edgecolor='white'),
           labeldistance=1.05)
   
    ax.pie(category_sizes, labels=categories, colors=category_colors, radius=0.7,
           wedgeprops=dict(width=0.4, edgecolor='white'),
           labeldistance=0.6, autopct='%1.1f%%', pctdistance=0.75)
   
    center_circle = plt.Circle((0, 0), 0.3, fc='white')
    ax.add_artist(center_circle)
 
    plt.axis('equal')
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Transcript Analysis Dashboard")
    st.title("Transcript Analysis Dashboard")

    tab1, tab2 = st.tabs(["Upload and Analyze", "View Saved Analysis"])

    with tab1:
        st.header("Upload and Analyze Transcripts")
        uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file)
            
            if st.button("Analyze Transcripts"):
                with st.spinner("Analyzing transcripts..."):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        results = list(executor.map(process_transcript, [row for index, row in data.iterrows()]))
                    
                    results = [result for result in results if result is not None]
                    results_df = pd.DataFrame(results)
                    
                    summary_json = summarize_sentiments(results_df)
                    openai_summary = summarize_with_openai(summary_json)
                    openai_data = json.loads(openai_summary)
                   
                    st.success("Analysis complete!")

                st.header("Analysis Results")

                st.subheader("Sentiment Distribution")
                sentiment_chart = create_sentiment_chart(openai_data['sentiment_distribution'])
                st.plotly_chart(sentiment_chart)

                st.subheader("Top Categories and Keywords")
                for category in openai_data['top_categories_and_keywords']:
                    st.write(f"**{category['category']}** ({category['percentage']:.2f}%)")
                    keywords_df = pd.DataFrame(category['top_keywords'])
                    st.dataframe(keywords_df)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top Strength Areas")
                    strengths_df = pd.DataFrame(openai_data['top_strength_areas'])
                    st.dataframe(strengths_df)
                    
                    # New: Strength Areas Chart
                    if not strengths_df.empty:
                        strength_chart = px.bar(strengths_df, x='area', y='frequency', title="Top Strength Areas")
                        st.plotly_chart(strength_chart, use_container_width=True)
                
                with col2:
                    st.subheader("Top Weak Areas")
                    weaknesses_df = pd.DataFrame(openai_data['top_weak_areas'])
                    st.dataframe(weaknesses_df)
                    
                    # New: Weak Areas Chart
                    if not weaknesses_df.empty:
                        weak_chart = px.bar(weaknesses_df, x='area', y='frequency', title="Top Weak Areas")
                        st.plotly_chart(weak_chart, use_container_width=True)

                st.subheader("Insights for Improvement")
                insights_df = pd.DataFrame(openai_data['insights_for_improvement'])
                st.dataframe(insights_df)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top Categories")
                    category_chart = create_category_chart(summary_json['category'])
                    st.plotly_chart(category_chart)
                with col2:
                    st.subheader("Topic Relevance")
                    topics_chart = create_topics_chart(openai_data['top_topics'])
                    st.plotly_chart(topics_chart)

                st.subheader("Categories and Keywords Distribution")
                fig = create_improved_nested_pie_chart(openai_data)
                st.pyplot(fig)

    with tab2:
        st.header("View Saved Analysis")
        try:
            saved_data = load_json_from_file('openai_summary.json')

            st.subheader("Sentiment Distribution")
            sentiment_chart = create_sentiment_chart(saved_data['sentiment_distribution'])
            st.plotly_chart(sentiment_chart)

            st.subheader("Top Categories and Keywords")
            if 'top_categories_and_keywords' in saved_data:
                for category in saved_data['top_categories_and_keywords']:
                    st.write(f"**{category['category']}** ({category['percentage']:.2f}%)")
                    keywords_df = pd.DataFrame(category['top_keywords'])
                    st.dataframe(keywords_df)
            else:
                st.warning("Categories and keywords data not found in saved analysis.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Strength Areas")
                if 'top_strength_areas' in saved_data:
                    strengths_df = pd.DataFrame(saved_data['top_strength_areas'])
                    st.dataframe(strengths_df)
                    
                    # New: Strength Areas Chart
                    if not strengths_df.empty:
                        strength_chart = px.bar(strengths_df, x='area', y='frequency', title="Top Strength Areas")
                        st.plotly_chart(strength_chart, use_container_width=True)
                else:
                    st.warning("Strength areas data not found in saved analysis.")
            
            with col2:
                st.subheader("Top Weak Areas")
                if 'top_weak_areas' in saved_data:
                    weaknesses_df = pd.DataFrame(saved_data['top_weak_areas'])
                    st.dataframe(weaknesses_df)
                    
                    # New: Weak Areas Chart
                    if not weaknesses_df.empty:
                        weak_chart = px.bar(weaknesses_df, x='area', y='frequency', title="Top Weak Areas")
                        st.plotly_chart(weak_chart, use_container_width=True)
                else:
                    st.warning("Weak areas data not found in saved analysis.")

            st.subheader("Insights for Improvement")
            if 'insights_for_improvement' in saved_data:
                insights_df = pd.DataFrame(saved_data['insights_for_improvement'])
                st.dataframe(insights_df)
            else:
                st.warning("Insights for improvement data not found in saved analysis.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Categories")
                if 'top_categories_and_keywords' in saved_data:
                    category_chart = create_category_chart(saved_data['top_categories_and_keywords'])
                    st.plotly_chart(category_chart)
                else:
                    st.warning("Category data not found in saved analysis.")
            with col2:
                st.subheader("Topic Relevance")
                if 'top_topics' in saved_data:
                    topics_chart = create_topics_chart(saved_data['top_topics'])
                    st.plotly_chart(topics_chart)
                else:
                    st.warning("Topic data not found in saved analysis.")

            st.subheader("Categories and Keywords Distribution")
            fig = create_improved_nested_pie_chart(saved_data)
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning("No saved analysis found. Please upload and analyze transcripts first.")

if __name__ == "__main__":
    main()