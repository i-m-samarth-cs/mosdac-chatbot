import streamlit as st
import json
import re
import datetime
from typing import Dict, List, Tuple, Optional
import time
import pandas as pd
from dataclasses import dataclass
import hashlib

# Configure page
st.set_page_config(
    page_title="🛰️ MOSDAC Chatbot",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-attractive UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 50%, #48ca61 100%);
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-content {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class ChatResponse:
    message: str
    confidence: float
    source: str
    context: str
    language: str = "en"

class KnowledgeGraph:
    """Simplified Knowledge Graph for ISRO/MOSDAC data"""
    
    def __init__(self):
        self.entities = {
            "satellites": ["INSAT-3D", "INSAT-3DR", "Megha-Tropiques", "SCATSAT-1", "INSAT-3A", "KALPANA-1"],
            "data_types": ["rainfall", "temperature", "humidity", "wind", "cloud_motion", "ocean_winds", "SST"],
            "regions": ["Maharashtra", "Gujarat", "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Telangana", "Kerala"],
            "time_periods": ["daily", "weekly", "monthly", "seasonal", "annual"],
            "formats": ["NetCDF", "HDF5", "GeoTIFF", "PNG", "JPG", "PDF"]
        }
        
        self.relationships = {
            "INSAT-3D": ["temperature", "humidity", "rainfall", "wind"],
            "INSAT-3DR": ["temperature", "humidity", "rainfall", "cloud_motion"],
            "SCATSAT-1": ["ocean_winds", "wind"],
            "Megha-Tropiques": ["rainfall", "humidity", "temperature"],
            "Maharashtra": ["rainfall", "temperature", "drought_monitoring"],
            "Gujarat": ["cyclone_tracking", "rainfall", "wind"],
        }
    
    def search_entities(self, query: str) -> List[str]:
        """Search for entities in the knowledge graph"""
        found_entities = []
        query_lower = query.lower()
        
        for category, entities in self.entities.items():
            for entity in entities:
                if entity.lower() in query_lower:
                    found_entities.append(f"{category}:{entity}")
        
        return found_entities

class RAGSystem:
    """Retrieval-Augmented Generation system for ISRO data"""
    
    def __init__(self):
        self.document_store = {
            "mosdac_overview": {
                "content": "MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) provides satellite data products for ocean, land, and atmosphere studies. It hosts data from various Indian satellites including INSAT-3D, INSAT-3DR, Megha-Tropiques, and SCATSAT-1.",
                "metadata": {"type": "overview", "confidence": 0.95}
            },
            "data_access": {
                "content": "To access MOSDAC data, users must register on the portal and log in. All data is free for academic and research purposes. Users can download various products in different formats including NetCDF, HDF5, and GeoTIFF.",
                "metadata": {"type": "procedure", "confidence": 0.9}
            },
            "satellite_data": {
                "content": "MOSDAC hosts data from Indian satellites: INSAT-3D provides temperature, humidity, and rainfall data. INSAT-3DR offers similar products with improved accuracy. SCATSAT-1 provides ocean wind measurements. Megha-Tropiques focuses on tropical weather and climate studies.",
                "metadata": {"type": "technical", "confidence": 0.92}
            },
            "geospatial_data": {
                "content": "MOSDAC provides region-specific data for all Indian states. Users can query data by geographical regions and time periods. Historical data is available from 2014 onwards for most satellites.",
                "metadata": {"type": "geospatial", "confidence": 0.88}
            }
        }
        
        # Hindi translations
        self.hindi_responses = {
            "mosdac_overview": "मोसडैक (मौसम विज्ञान और समुद्री उपग्रह डेटा संग्रह केंद्र) समुद्र, भूमि और वायुमंडल अध्ययन के लिए उपग्रह डेटा उत्पाद प्रदान करता है।",
            "data_access": "मोसडैक डेटा तक पहुंचने के लिए, उपयोगकर्ताओं को पोर्टल पर पंजीकरण करना होगा। सभी डेटा शैक्षणिक और अनुसंधान उद्देश्यों के लिए निःशुल्क है।",
            "satellite_data": "मोसडैक भारतीय उपग्रहों से डेटा होस्ट करता है: INSAT-3D तापमान, आर्द्रता और वर्षा डेटा प्रदान करता है।"
        }
    
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant documents based on query"""
        results = []
        query_lower = query.lower()
        
        for doc_id, doc_data in self.document_store.items():
            content = doc_data["content"].lower()
            
            # Simple relevance scoring
            score = 0
            query_words = query_lower.split()
            for word in query_words:
                if word in content:
                    score += 1
            
            if score > 0:
                results.append({
                    "id": doc_id,
                    "content": doc_data["content"],
                    "score": score / len(query_words),
                    "metadata": doc_data["metadata"]
                })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

class ISROChatbot:
    """Main chatbot class combining KG and RAG"""
    
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.rag = RAGSystem()
        self.conversation_history = []
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        hindi_chars = re.findall(r'[\u0900-\u097F]', text)
        return "hi" if len(hindi_chars) > 0 else "en"
    
    def extract_geospatial_info(self, query: str) -> Dict:
        """Extract geospatial and temporal information from query"""
        info = {"region": None, "time_period": None, "data_type": None}
        
        # Extract regions
        for region in self.kg.entities["regions"]:
            if region.lower() in query.lower():
                info["region"] = region
                break
        
        # Extract time periods
        time_patterns = {
            r'(\d{4})': 'year',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)': 'month',
            r'(daily|weekly|monthly|seasonal|annual)': 'period'
        }
        
        for pattern, period_type in time_patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                info["time_period"] = f"{period_type}:{match.group(1)}"
                break
        
        # Extract data types
        for data_type in self.kg.entities["data_types"]:
            if data_type.lower() in query.lower():
                info["data_type"] = data_type
                break
        
        return info
    
    def calculate_confidence(self, query: str, response: str, entities: List[str]) -> float:
        """Calculate confidence score for response"""
        base_confidence = 0.7
        
        # Boost confidence if entities are found
        if entities:
            base_confidence += 0.1 * len(entities)
        
        # Boost confidence if geospatial info is found
        geo_info = self.extract_geospatial_info(query)
        if any(geo_info.values()):
            base_confidence += 0.1
        
        # Boost confidence for exact matches
        if any(keyword in query.lower() for keyword in ["mosdac", "satellite", "data", "isro"]):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def generate_response(self, query: str) -> ChatResponse:
        """Generate response using hybrid KG + RAG approach"""
        
        # Detect language
        language = self.detect_language(query)
        
        # Search knowledge graph
        entities = self.kg.search_entities(query)
        
        # Extract geospatial information
        geo_info = self.extract_geospatial_info(query)
        
        # Retrieve relevant documents
        documents = self.rag.retrieve_documents(query)
        
        # Generate response based on findings
        if documents:
            primary_doc = documents[0]
            response_text = primary_doc["content"]
            confidence = self.calculate_confidence(query, response_text, entities)
            source = primary_doc["id"]
            
            # Add geospatial context if available
            if geo_info["region"]:
                response_text += f"\n\nFor {geo_info['region']} region, "
                if geo_info["time_period"]:
                    response_text += f"data from {geo_info['time_period']} is available through MOSDAC portal."
                else:
                    response_text += "historical and real-time data is available through MOSDAC portal."
            
            # Translate to Hindi if needed
            if language == "hi" and source in self.rag.hindi_responses:
                response_text = self.rag.hindi_responses[source]
            
            return ChatResponse(
                message=response_text,
                confidence=confidence,
                source=source,
                context=f"Found {len(entities)} relevant entities, {len(documents)} documents",
                language=language
            )
        
        # Fallback response
        fallback_msg = "I can help you with MOSDAC data queries. Try asking about satellite data, data access, or specific regions and time periods."
        if language == "hi":
            fallback_msg = "मैं आपको मोसडैक डेटा के बारे में जानकारी दे सकता हूं। उपग्रह डेटा, डेटा पहुंच, या विशिष्ट क्षेत्रों के बारे में पूछें।"
        
        return ChatResponse(
            message=fallback_msg,
            confidence=0.3,
            source="fallback",
            context="No specific documents found",
            language=language
        )

# Initialize chatbot
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = ISROChatbot()
    st.session_state.conversation = []

# Main UI
st.markdown('<div class="main-header"><h1>🛰️ MOSDAC Chatbot</h1><p>Advanced AI Assistant for MOSDAC & Satellite Data</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content"><h3>🚀 Features</h3></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>🧠 Hybrid AI</h4>
        <p>Knowledge Graph + RAG System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>🌍 Geospatial Intelligence</h4>
        <p>Region & Time-based Queries</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>🌐 Multilingual</h4>
        <p>English & Hindi Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Smart Analytics</h4>
        <p>Confidence Scoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick examples
    st.markdown("### 💡 Try these queries:")
    examples = [
        "What is MOSDAC?",
        "Maharashtra rainfall data 2022",
        "How to access satellite data?",
        "INSAT-3D temperature data",
        "मोसडैक क्या है?",
        "Gujarat cyclone tracking data"
    ]
    
    for example in examples:
        if st.button(example, key=f"example_{hash(example)}"):
            st.session_state.current_query = example

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat input
    if 'current_query' in st.session_state:
        query = st.session_state.current_query
        del st.session_state.current_query
    else:
        query = st.text_input("🔍 Ask me anything about ISRO/MOSDAC:", placeholder="e.g., Maharashtra images for March 2022")
    
    # Process query
    if query:
        with st.spinner("🤖 Processing your query..."):
            response = st.session_state.chatbot.generate_response(query)
            st.session_state.conversation.append({"query": query, "response": response})
    
    # Display conversation
    for i, conv in enumerate(reversed(st.session_state.conversation[-5:])):  # Show last 5 conversations
        st.markdown(f'<div class="user-message">🧑‍💻 {conv["query"]}</div>', unsafe_allow_html=True)
        
        # Bot response with confidence bar
        st.markdown(f'<div class="bot-message">🤖 {conv["response"].message}</div>', unsafe_allow_html=True)
        
        # Confidence indicator
        confidence_width = conv["response"].confidence * 100
        st.markdown(f'''
        <div style="margin: 0.5rem 0;">
            <small>Confidence: {conv["response"].confidence:.1%}</small>
            <div style="background: #e0e0e0; height: 6px; border-radius: 3px; margin: 0.2rem 0;">
                <div style="background: linear-gradient(90deg, #ff6b6b 0%, #feca57 50%, #48ca61 100%); 
                           height: 6px; border-radius: 3px; width: {confidence_width}%;"></div>
            </div>
            <small>Source: {conv["response"].source} | Context: {conv["response"].context}</small>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📈 Chat Analytics")
    
    if st.session_state.conversation:
        # Calculate average confidence
        avg_confidence = sum(conv["response"].confidence for conv in st.session_state.conversation) / len(st.session_state.conversation)
        
        st.metric("Average Confidence", f"{avg_confidence:.1%}")
        st.metric("Total Queries", len(st.session_state.conversation))
        
        # Language distribution
        languages = [conv["response"].language for conv in st.session_state.conversation]
        lang_counts = {"English": languages.count("en"), "Hindi": languages.count("hi")}
        
        st.markdown("**Language Usage:**")
        for lang, count in lang_counts.items():
            st.write(f"{lang}: {count}")
    
    # Clear conversation
    if st.button("🗑️ Clear Chat"):
        st.session_state.conversation = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🛰️ MOSDAC Chatbot | By Team Space Gladiators</p>
    <p>Features: Knowledge Graph • RAG System • Geospatial Intelligence • Multilingual Support</p>
</div>
""", unsafe_allow_html=True)