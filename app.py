# Personal Knowledge Assistant with Neo4j and Visualization
# A system that ingests documents, creates knowledge graphs, and provides interactive visualization
 
import os
import json
import logging
from typing import Dict, List, Tuple, Set, Optional 
from dataclasses import dataclass, asdict
from pathlib import Path 
import re
from collections import defaultdict

# Graph database
from neo4j import GraphDatabase
import neo4j

# Document processing
import PyPDF2
from docx import Document as DocxDocument
import markdown

# NLP
import spacy

# LLM Integration
from google import genai
from google.genai import types

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd

# Web interface
import streamlit as st
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    name: str
    type: str  # PERSON, ORG, CONCEPT, LOCATION, EVENT, etc.
    description: str = ""
    source_documents: Set[str] = None
    
    def __post_init__(self):
        if self.source_documents is None:
            self.source_documents = set()

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source: str
    target: str
    relation_type: str
    description: str = ""
    confidence: float = 1.0
    source_documents: Set[str] = None
    
    def __post_init__(self):
        if self.source_documents is None:
            self.source_documents = set()

class Neo4jKnowledgeGraph:
    """Neo4j-based knowledge graph implementation"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j database")
            
            # Create constraints and indexes
            self._create_constraints()
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.info("Please ensure Neo4j is running on bolt://localhost:7687")
            logger.info("To start Neo4j: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
            raise
    
    def _create_constraints(self):
        """Create necessary constraints and indexes"""
        with self.driver.session() as session:
            # Create uniqueness constraints
            constraints = [
                "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                # Create indexes for better performance
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX entity_name_text IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.debug(f"Constraint/Index creation: {e}")
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
    
    def add_entity(self, entity: Entity):
        """Add or update an entity in Neo4j"""
        with self.driver.session() as session:
            query = """
            MERGE (e:Entity {name: $name})
            SET e.type = $type,
                e.description = $description,
                e.source_documents = $source_documents,
                e.updated_at = datetime()
            RETURN e
            """
            
            result = session.run(query, 
                name=entity.name,
                type=entity.type,
                description=entity.description or "",
                source_documents=list(entity.source_documents)
            )
            return result.single()
    
    def add_relationship(self, relationship: Relationship):
        """Add a relationship between entities"""
        with self.driver.session() as session:
            # First ensure both entities exist
            self._ensure_entity_exists(session, relationship.source)
            self._ensure_entity_exists(session, relationship.target)
            
            # Sanitize relationship type for Cypher (remove spaces, special chars)
            safe_rel_type = ''.join(c for c in relationship.relation_type.upper() if c.isalnum() or c == '_')
            if not safe_rel_type:
                safe_rel_type = 'RELATES_TO'
            
            # Create relationship with dynamic relationship type
            query = f"""
            MATCH (source:Entity {{name: $source_name}})
            MATCH (target:Entity {{name: $target_name}})
            MERGE (source)-[r:{safe_rel_type}]->(target)
            SET r.description = $description,
                r.confidence = $confidence,
                r.source_documents = $source_documents,
                r.updated_at = datetime()
            RETURN r
            """
            
            result = session.run(query,
                source_name=relationship.source,
                target_name=relationship.target,
                description=relationship.description or "",
                confidence=relationship.confidence,
                source_documents=list(relationship.source_documents)
            )
            return result.single()
    
    def _ensure_entity_exists(self, session, entity_name: str):
        """Ensure an entity exists (create minimal version if not)"""
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = 'UNKNOWN', e.created_at = datetime()
        RETURN e
        """
        session.run(query, name=entity_name)
    
    def add_document(self, document_id: str, metadata: Dict):
        """Add document metadata to the graph"""
        with self.driver.session() as session:
            query = """
            MERGE (d:Document {id: $doc_id})
            SET d.filename = $filename,
                d.filepath = $filepath,
                d.file_type = $file_type,
                d.content_length = $content_length,
                d.processed_at = datetime()
            RETURN d
            """
            
            result = session.run(query,
                doc_id=document_id,
                filename=metadata.get('filename', ''),
                filepath=metadata.get('filepath', ''),
                file_type=metadata.get('file_type', ''),
                content_length=metadata.get('content_length', 0)
            )
            return result.single()
    
    def link_entity_to_document(self, entity_name: str, document_id: str):
        """Link an entity to its source document"""
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity {name: $entity_name})
            MATCH (d:Document {id: $doc_id})
            MERGE (e)-[:MENTIONED_IN]->(d)
            """
            session.run(query, entity_name=entity_name, doc_id=document_id)
    
    def find_related_entities(self, entity_name: str, max_depth: int = 2, limit: int = 50) -> List[Dict]:
        """Find entities related to the given entity"""
        with self.driver.session() as session:
            query = """
            MATCH path = (start:Entity {name: $entity_name})-[*1..$max_depth]-(related:Entity)
            WHERE start <> related
            RETURN DISTINCT related.name as name, 
                   related.type as type, 
                   related.description as description,
                   length(path) as distance
            ORDER BY distance, related.name
            LIMIT $limit
            """
            
            result = session.run(query, 
                entity_name=entity_name, 
                max_depth=max_depth, 
                limit=limit
            )
            
            return [dict(record) for record in result]
    
    def get_entity_details(self, entity_name: str) -> Optional[Dict]:
        """Get detailed information about an entity"""
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity {name: $entity_name})
            OPTIONAL MATCH (e)-[r]->(target:Entity)
            OPTIONAL MATCH (e)<-[r2]-(source:Entity)
            OPTIONAL MATCH (e)-[:MENTIONED_IN]->(d:Document)
            RETURN e,
                   collect(DISTINCT {target: target.name, relation: type(r), description: r.description}) as outgoing_relations,
                   collect(DISTINCT {source: source.name, relation: type(r2), description: r2.description}) as incoming_relations,
                   collect(DISTINCT d.filename) as source_documents
            """
            
            result = session.run(query, entity_name=entity_name)
            record = result.single()
            
            if record:
                entity = record['e']
                return {
                    'name': entity['name'],
                    'type': entity['type'],
                    'description': entity.get('description', ''),
                    'outgoing_relations': [rel for rel in record['outgoing_relations'] if rel['target']],
                    'incoming_relations': [rel for rel in record['incoming_relations'] if rel['source']],
                    'source_documents': record['source_documents']
                }
            return None
    
    def search_entities(self, query: str, limit: int = 20) -> List[Dict]:
        """Search entities by name or description"""
        with self.driver.session() as session:
            cypher_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($search_term) 
               OR toLower(coalesce(e.description, '')) CONTAINS toLower($search_term)
               OR any(word IN split(toLower($search_term), ' ') WHERE toLower(e.name) CONTAINS word)
            RETURN e.name as name, e.type as type, coalesce(e.description, '') as description
            ORDER BY 
                CASE 
                    WHEN toLower(e.name) = toLower($search_term) THEN 1 
                    WHEN toLower(e.name) CONTAINS toLower($search_term) THEN 2
                    ELSE 3 
                END,
                size(e.name)
            LIMIT $limit
            """
            
            result = session.run(cypher_query, search_term=query, limit=limit)
            return [dict(record) for record in result]
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the knowledge graph"""
        with self.driver.session() as session:
            stats_query = """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r]-()
            OPTIONAL MATCH (d:Document)
            RETURN 
                count(DISTINCT e) as total_entities,
                count(DISTINCT r) as total_relationships,
                count(DISTINCT d) as total_documents
            """
            
            result = session.run(stats_query)
            record = result.single()
            
            # Get entity type counts
            type_counts_query = """
            MATCH (e:Entity)
            RETURN e.type as type, count(e) as count
            ORDER BY count DESC
            """
            
            type_result = session.run(type_counts_query)
            type_counts = {record['type']: record['count'] for record in type_result}
            
            return {
                'total_entities': record['total_entities'],
                'total_relationships': record['total_relationships'] // 2,  # Undirected count
                'total_documents': record['total_documents'],
                'entity_types': type_counts
            }
    
    def get_subgraph(self, entity_names: List[str] = None, max_nodes: int = 100) -> Dict:
        """Get a subgraph for visualization"""
        with self.driver.session() as session:
            if entity_names:
                # Get subgraph around specific entities
                query = """
                MATCH (e:Entity)
                WHERE e.name IN $entity_names
                OPTIONAL MATCH path = (e)-[*1..2]-(connected:Entity)
                WITH collect(DISTINCT e) + 
                     collect(DISTINCT connected) as all_entities
                UNWIND all_entities as entity
                WITH collect(DISTINCT entity)[0..$max_nodes] as limited_entities
                UNWIND limited_entities as e1
                OPTIONAL MATCH (e1)-[r]-(e2)
                WHERE e2 IN limited_entities
                RETURN 
                    collect(DISTINCT {id: e1.name, name: e1.name, type: e1.type, description: coalesce(e1.description, '')}) as nodes,
                    collect(DISTINCT {source: e1.name, target: e2.name, type: type(r), description: coalesce(r.description, '')}) as edges
                """
                result = session.run(query, entity_names=entity_names, max_nodes=max_nodes)
            else:
                # Get general subgraph
                query = """
                MATCH (e:Entity)-[r]-(e2:Entity)
                WITH e, r, e2
                LIMIT $max_nodes
                RETURN 
                    collect(DISTINCT {id: e.name, name: e.name, type: e.type, description: coalesce(e.description, '')}) +
                    collect(DISTINCT {id: e2.name, name: e2.name, type: e2.type, description: coalesce(e2.description, '')}) as nodes,
                    collect({source: e.name, target: e2.name, type: type(r), description: coalesce(r.description, '')}) as edges
                """
                result = session.run(query, max_nodes=max_nodes)
            
            record = result.single()
            if record:
                # Remove duplicates from nodes
                unique_nodes = {}
                for node in record['nodes'] or []:
                    if node and node.get('id'):
                        unique_nodes[node['id']] = node
                
                # Filter valid edges
                valid_edges = []
                for edge in record['edges'] or []:
                    if edge and edge.get('source') and edge.get('target'):
                        valid_edges.append(edge)
                
                return {
                    'nodes': list(unique_nodes.values()),
                    'edges': valid_edges
                }
            
            return {'nodes': [], 'edges': []}
    
    def clear_database(self):
        """Clear all data from the database (use with caution!)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")

class DocumentProcessor:
    """Enhanced document processor"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def process_file(self, file_path: str) -> Dict:
        """Process a file and extract text content with enhanced metadata"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = ""
        metadata = {
            "filename": file_path.name,
            "filepath": str(file_path),
            "file_type": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size,
            "modified_time": file_path.stat().st_mtime
        }
        
        try:
            if file_path.suffix.lower() == '.pdf':
                content = self._extract_pdf_text(file_path)
            elif file_path.suffix.lower() == '.docx':
                content = self._extract_docx_text(file_path)
            elif file_path.suffix.lower() == '.txt':
                content = self._extract_txt_text(file_path)
            elif file_path.suffix.lower() == '.md':
                content = self._extract_markdown_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            metadata["content_length"] = len(content)
            metadata["word_count"] = len(content.split())
            
            return {"content": content, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {"content": "", "metadata": metadata}
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files"""
        doc = DocxDocument(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
            text = markdown.markdown(md_content)
            text = re.sub('<[^<]+?>', '', text)
            return text

class EnhancedEntityExtractor:
    """Enhanced entity extractor with better LLM prompting"""
    
    def __init__(self, gemini_api_key: str):
        self.client = genai.Client(api_key=gemini_api_key)
        self.model = "gemini-2.0-flash"
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not available")
            self.nlp = None
    
    def extract_entities_and_relations(self, text: str, document_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships with improved methods"""
        
        # Method 1: SpaCy NER for basic entities
        spacy_entities = self._extract_spacy_entities(text, document_id) if self.nlp else []
        
        # Method 2: Enhanced LLM extraction
        llm_entities, llm_relations = self._extract_llm_entities_relations(text, document_id)
        
        # Merge and deduplicate
        all_entities = self._merge_entities(spacy_entities + llm_entities)
        
        return all_entities, llm_relations
    
    def _extract_spacy_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if len(ent.text.strip()) > 1:  # Filter out single characters
                entity = Entity(
                    name=ent.text.strip(),
                    type=ent.label_,
                    description=f"{ent.label_} entity from {document_id}",
                    source_documents={document_id}
                )
                entities.append(entity)
        
        return entities
    
    def _extract_llm_entities_relations(self, text: str, document_id: str) -> Tuple[List[Entity], List[Relationship]]:
        """Enhanced LLM extraction with better prompting"""
        
        chunks = self._split_text(text, max_length=2500)
        all_entities = []
        all_relations = []
        
        for i, chunk in enumerate(chunks):
            try:
                entities, relations = self._process_chunk_with_llm(chunk, document_id, chunk_id=i)
                all_entities.extend(entities)
                all_relations.extend(relations)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
        
        return all_entities, all_relations
    
    def _process_chunk_with_llm(self, text: str, document_id: str, chunk_id: int = 0) -> Tuple[List[Entity], List[Relationship]]:
        """Process text chunk with enhanced LLM prompting"""
        
        prompt = f"""
        You are an expert knowledge graph builder. Analyze the following text and extract meaningful entities and their relationships.

        Instructions:
        1. Extract IMPORTANT entities only - focus on key concepts, people, organizations, locations, events, and domain-specific terms
        2. Avoid generic words like "system", "method", "way", "thing" unless they're specifically important
        3. For each entity, provide a clear type and brief description
        4. Extract meaningful relationships between entities
        5. Focus on factual, explicit relationships mentioned in the text

        Entity Types to consider:
        - PERSON: People, authors, historical figures
        - ORGANIZATION: Companies, institutions, groups
        - LOCATION: Places, countries, cities
        - CONCEPT: Ideas, theories, methodologies, technologies
        - EVENT: Specific events, meetings, incidents
        - PRODUCT: Tools, software, books, documents
        - DATE: Specific dates, time periods
        - OTHER: Other important domain-specific entities

        Text to analyze:
        {text}

        Return ONLY a valid JSON object in this exact format:
        {{
            "entities": [
                {{"name": "Entity Name", "type": "TYPE", "description": "Brief description of what this entity is"}}
            ],
            "relationships": [
                {{"source": "Entity1", "target": "Entity2", "relation": "relationship_type", "description": "Brief description of the relationship"}}
            ]
        }}

        Important: 
        - Only include entities that are genuinely important to understanding the content
        - Ensure all relationship sources and targets refer to entities in the entities list
        - Use clear, descriptive relationship types like "works_at", "located_in", "develops", "influences", etc.
        """
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            
            response_text = response.text.strip()
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                data = json.loads(json_text)
                
                entities = []
                relations = []
                
                # Process entities
                for ent_data in data.get('entities', []):
                    if len(ent_data.get('name', '').strip()) > 1:  # Filter short names
                        entity = Entity(
                            name=ent_data['name'].strip(),
                            type=ent_data.get('type', 'OTHER'),
                            description=ent_data.get('description', ''),
                            source_documents={document_id}
                        )
                        entities.append(entity)
                
                # Process relationships
                entity_names = {e.name for e in entities}
                for rel_data in data.get('relationships', []):
                    source = rel_data.get('source', '').strip()
                    target = rel_data.get('target', '').strip()
                    
                    # Only include relationships where both entities exist
                    if source in entity_names and target in entity_names and source != target:
                        relation = Relationship(
                            source=source,
                            target=target,
                            relation_type=rel_data.get('relation', 'related_to'),
                            description=rel_data.get('description', ''),
                            source_documents={document_id}
                        )
                        relations.append(relation)
                
                return entities, relations
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
        
        return [], []
    
    def _split_text(self, text: str, max_length: int = 2500) -> List[str]:
        """Split text into manageable chunks"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge similar entities with improved logic"""
        merged = {}
        
        for entity in entities:
            # Normalize entity name for comparison
            key = entity.name.lower().strip()
            
            # Simple similarity check for variations
            found_match = False
            for existing_key in merged.keys():
                if self._are_similar_entities(key, existing_key):
                    # Merge with existing
                    merged[existing_key].source_documents.update(entity.source_documents)
                    if len(entity.description) > len(merged[existing_key].description):
                        merged[existing_key].description = entity.description
                    found_match = True
                    break
            
            if not found_match:
                merged[key] = entity
        
        return list(merged.values())
    
    def _are_similar_entities(self, name1: str, name2: str) -> bool:
        """Check if two entity names are similar enough to merge"""
        # Simple similarity checks
        if name1 == name2:
            return True
        
        # Check if one is contained in the other
        if name1 in name2 or name2 in name1:
            return True
        
        # Check for common abbreviations or variations
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        # If significant overlap in words
        if len(words1 & words2) / max(len(words1), len(words2)) > 0.7:
            return True
        
        return False

class KnowledgeGraphVisualizer:
    """Advanced visualization for knowledge graphs"""
    
    def __init__(self):
        self.color_map = {
            'PERSON': '#FF6B6B',
            'ORGANIZATION': '#4ECDC4', 
            'LOCATION': '#45B7D1',
            'CONCEPT': '#96CEB4',
            'EVENT': '#FECA57',
            'PRODUCT': '#FF9FF3',
            'DATE': '#FD79A8',
            'OTHER': '#DDA0DD'
        }
    
    def create_interactive_graph(self, graph_data: Dict, title: str = "Knowledge Graph") -> go.Figure:
        """Create an interactive Plotly graph visualization"""
        
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        if not nodes:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(text="No data to visualize", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Create NetworkX graph for layout
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # Add edges
        for edge in edges:
            if edge['source'] in G.nodes and edge['target'] in G.nodes:
                G.add_edge(edge['source'], edge['target'], **edge)
        
        # Generate layout
        if len(G.nodes) > 100:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = nx.spring_layout(G, k=3, iterations=100)
        
        # Prepare node traces
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            marker=dict(
                size=15,
                color=[],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Entity Type",
                    x=1.02
                ),
                line=dict(width=2, color='black')
            ),
            text=[],
            textposition="middle center",
            textfont=dict(size=10),
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Type: %{customdata[1]}<br>' +
                         'Description: %{customdata[2]}<br>' +
                         '<extra></extra>',
            customdata=[]
        )
        
        # Add node coordinates and info
        for node in nodes:
            node_id = node['id']
            if node_id in pos:
                x, y = pos[node_id]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                
                # Color by type
                node_type = node.get('type', 'OTHER')
                color_idx = list(self.color_map.keys()).index(node_type) if node_type in self.color_map else 0
                node_trace['marker']['color'] += tuple([color_idx])
                
                # Node label (truncate if too long)
                label = node['name'][:15] + '...' if len(node['name']) > 15 else node['name']
                node_trace['text'] += tuple([label])
                
                # Hover info
                node_trace['customdata'] += tuple([[
                    node['name'], 
                    node.get('type', 'Unknown'),
                    node.get('description', 'No description')[:100] + '...' if len(node.get('description', '')) > 100 else node.get('description', 'No description')
                ]])
        
        # Prepare edge traces
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1, color='#888'),
            mode='lines',
            hoverinfo='none'
        )
        
        # Add edge coordinates
        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            
            if source_id in pos and target_id in pos:
                x0, y0 = pos[source_id]
                x1, y1 = pos[target_id]
                
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text=title, x=0.5),
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes for details. Drag to move.",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='gray', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def create_entity_type_chart(self, stats: Dict) -> go.Figure:
        """Create a pie chart of entity types"""
        entity_types = stats.get('entity_types', {})
        
        if not entity_types:
            fig = go.Figure()
            fig.add_annotation(text="No entity data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        colors = [self.color_map.get(entity_type, '#DDA0DD') for entity_type in entity_types.keys()]
        
        fig = go.Figure(data=[go.Pie(
            labels=list(entity_types.keys()),
            values=list(entity_types.values()),
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent+value',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Entity Types Distribution",
            annotations=[dict(text='Entities', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def create_network_metrics_chart(self, graph_data: Dict) -> go.Figure:
        """Create network analysis metrics visualization"""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        if not nodes or not edges:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for network analysis", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Build NetworkX graph for analysis
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'])
        for edge in edges:
            if edge['source'] in G.nodes and edge['target'] in G.nodes:
                G.add_edge(edge['source'], edge['target'])
        
        # Calculate metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Get top nodes by degree centrality
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:15]
        
        node_names = [node[0] for node in top_nodes]
        degree_values = [degree_centrality[node] for node in node_names]
        betweenness_values = [betweenness_centrality[node] for node in node_names]
        closeness_values = [closeness_centrality[node] for node in node_names]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Degree Centrality', 'Betweenness Centrality', 
                          'Closeness Centrality', 'Network Overview'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Degree centrality
        fig.add_trace(
            go.Bar(x=node_names, y=degree_values, name="Degree", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Betweenness centrality
        fig.add_trace(
            go.Bar(x=node_names, y=betweenness_values, name="Betweenness", marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Closeness centrality
        fig.add_trace(
            go.Bar(x=node_names, y=closeness_values, name="Closeness", marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Network overview table
        metrics = [
            ["Total Nodes", len(G.nodes)],
            ["Total Edges", len(G.edges)],
            ["Density", f"{nx.density(G):.3f}"],
            ["Connected Components", nx.number_connected_components(G)],
            ["Average Clustering", f"{nx.average_clustering(G):.3f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=["Metric", "Value"], fill_color='lightblue'),
                cells=dict(values=list(zip(*metrics)), fill_color='white')
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Network Analysis Metrics", showlegend=False)
        fig.update_xaxes(tickangle=45)
        
        return fig

class StreamlitInterface:
    """Streamlit web interface for the Knowledge Assistant"""
    
    def __init__(self, assistant):
        self.assistant = assistant
        st.set_page_config(
            page_title="Personal Knowledge Assistant",
            page_icon="🧠",
            layout="wide"
        )
    
    def render(self):
        """Render the main Streamlit interface"""
        st.title("🧠 Personal Knowledge Assistant")
        st.markdown("*Build and explore your personal knowledge graph*")
        
        # Sidebar
        with st.sidebar:
            st.header("📊 Knowledge Base Stats")
            try:
                stats = self.assistant.get_statistics()
                st.metric("Entities", stats['total_entities'])
                st.metric("Relationships", stats['total_relationships'])
                st.metric("Documents", stats['total_documents'])
                
                if stats['entity_types']:
                    st.subheader("Entity Types")
                    for entity_type, count in stats['entity_types'].items():
                        st.write(f"**{entity_type}**: {count}")
            except Exception as e:
                st.error(f"Error loading stats: {e}")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Document Ingestion", 
            "🔍 Search & Query", 
            "📊 Visualization", 
            "🌐 Graph Explorer",
            "💬 AI Assistant"
        ])
        
        with tab1:
            self._render_ingestion_tab()
        
        with tab2:
            self._render_search_tab()
        
        with tab3:
            self._render_visualization_tab()
        
        with tab4:
            self._render_graph_explorer_tab()
        
        with tab5:
            self._render_ai_assistant_tab()
    
    def _render_ingestion_tab(self):
        """Render document ingestion interface"""
        st.header("📁 Document Ingestion")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_files = st.file_uploader(
                "Upload documents to add to your knowledge base",
                type=['pdf', 'docx', 'txt', 'md'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Process Documents", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_files = len(uploaded_files)
                    processed = 0
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        try:
                            # Save uploaded file temporarily
                            temp_path = f"temp_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Process the file
                            success = self.assistant.ingest_document(temp_path)
                            
                            if success:
                                processed += 1
                                st.success(f"✅ Processed {uploaded_file.name}")
                            else:
                                st.error(f"❌ Failed to process {uploaded_file.name}")
                            
                            # Clean up temp file
                            os.remove(temp_path)
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                        
                        progress_bar.progress((i + 1) / total_files)
                    
                    status_text.text(f"Completed! Processed {processed}/{total_files} files")
                    
                    # Refresh the page to update stats
                    if processed > 0:
                        st.rerun()
            
            # Directory ingestion
            st.subheader("Batch Processing")
            directory_path = st.text_input("Or enter a directory path to process all documents:")
            
            if st.button("Process Directory") and directory_path:
                if os.path.exists(directory_path):
                    with st.spinner("Processing directory..."):
                        self.assistant.ingest_directory(directory_path)
                    st.success("Directory processed!")
                    st.rerun()
                else:
                    st.error("Directory not found!")
        
        with col2:
            st.subheader("💡 Tips")
            st.info("""
            **Supported formats:**
            - PDF documents
            - Word documents (.docx)
            - Text files (.txt)
            - Markdown files (.md)
            
            **Best practices:**
            - Use descriptive filenames
            - Ensure documents have clear structure
            - Include context-rich content
            """)
    
    def _render_search_tab(self):
        """Render search and query interface"""
        st.header("🔍 Search & Query")
        
        # Search entities
        search_query = st.text_input("Search entities in your knowledge base:")
        
        if search_query:
            try:
                results = self.assistant.search_knowledge_base(search_query)
                
                if results:
                    st.write(f"Found {len(results)} matching entities:")
                    
                    for result in results:
                        with st.expander(f"🏷️ {result['entity_name']} ({result['entity_type']})"):
                            st.write(f"**Type:** {result['entity_type']}")
                            st.write(f"**Description:** {result['description']}")
                            st.write(f"**Source Documents:** {', '.join(result['source_documents'])}")
                            
                            # Show related entities
                            if st.button(f"Show related entities", key=f"related_{result['entity_name']}"):
                                related = self.assistant.knowledge_graph.find_related_entities(result['entity_name'])
                                if related:
                                    st.write("**Related entities:**")
                                    for rel in related[:10]:
                                        st.write(f"- {rel['name']} ({rel['type']}) - Distance: {rel['distance']}")
                else:
                    st.write("No matching entities found.")
                    
            except Exception as e:
                st.error(f"Search error: {e}")
        
        # Entity details viewer
        st.subheader("Entity Details")
        entity_name = st.text_input("Enter entity name for detailed view:")
        
        if entity_name:
            try:
                details = self.assistant.knowledge_graph.get_entity_details(entity_name)
                
                if details:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Name:** {details['name']}")
                        st.write(f"**Type:** {details['type']}")
                        st.write(f"**Description:** {details['description']}")
                        st.write(f"**Source Documents:** {', '.join(details['source_documents'])}")
                    
                    with col2:
                        if details['outgoing_relations']:
                            st.write("**Outgoing Relationships:**")
                            for rel in details['outgoing_relations']:
                                st.write(f"- {rel['relation']} → {rel['target']}")
                        
                        if details['incoming_relations']:
                            st.write("**Incoming Relationships:**")
                            for rel in details['incoming_relations']:
                                st.write(f"- {rel['source']} → {rel['relation']}")
                else:
                    st.write("Entity not found.")
                    
            except Exception as e:
                st.error(f"Error retrieving entity details: {e}")
    
    def _render_visualization_tab(self):
        """Render visualization interface"""
        st.header("📊 Knowledge Graph Visualization")
        
        try:
            stats = self.assistant.get_statistics()
            visualizer = KnowledgeGraphVisualizer()
            
            # Entity types chart
            if stats['entity_types']:
                st.subheader("Entity Types Distribution")
                fig_pie = visualizer.create_entity_type_chart(stats)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Network metrics
            st.subheader("Network Analysis")
            
            # Get sample graph data for analysis
            graph_data = self.assistant.knowledge_graph.get_subgraph(max_nodes=100)
            
            if graph_data['nodes']:
                fig_metrics = visualizer.create_network_metrics_chart(graph_data)
                st.plotly_chart(fig_metrics, use_container_width=True)
            else:
                st.info("Add more documents to see network analysis.")
                
        except Exception as e:
            st.error(f"Visualization error: {e}")
    
    def _render_graph_explorer_tab(self):
        """Render interactive graph explorer"""
        st.header("🌐 Interactive Graph Explorer")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Filter Options")
            
            # Node limit
            max_nodes = st.slider("Max nodes to display", 10, 200, 50)
            
            # Entity filter
            try:
                stats = self.assistant.get_statistics()
                entity_types = list(stats.get('entity_types', {}).keys())
                
                if entity_types:
                    selected_types = st.multiselect(
                        "Filter by entity type:",
                        entity_types,
                        default=entity_types
                    )
                else:
                    selected_types = []
                
                # Specific entities
                focus_entities = st.text_area(
                    "Focus on specific entities (one per line):",
                    placeholder="entity1\nentity2\nentity3"
                )
                
                focus_list = [e.strip() for e in focus_entities.split('\n') if e.strip()] if focus_entities else None
                
            except Exception as e:
                st.error(f"Error loading filter options: {e}")
                selected_types = []
                focus_list = None
        
        with col1:
            try:
                # Get graph data
                if focus_list:
                    graph_data = self.assistant.knowledge_graph.get_subgraph(
                        entity_names=focus_list, 
                        max_nodes=max_nodes
                    )
                else:
                    graph_data = self.assistant.knowledge_graph.get_subgraph(max_nodes=max_nodes)
                
                # Filter by entity types if specified
                if selected_types and graph_data['nodes']:
                    filtered_nodes = [
                        node for node in graph_data['nodes'] 
                        if node.get('type') in selected_types
                    ]
                    
                    # Keep only edges between filtered nodes
                    filtered_node_ids = {node['id'] for node in filtered_nodes}
                    filtered_edges = [
                        edge for edge in graph_data['edges']
                        if edge['source'] in filtered_node_ids and edge['target'] in filtered_node_ids
                    ]
                    
                    graph_data = {
                        'nodes': filtered_nodes,
                        'edges': filtered_edges
                    }
                
                if graph_data['nodes']:
                    visualizer = KnowledgeGraphVisualizer()
                    fig = visualizer.create_interactive_graph(graph_data, "Interactive Knowledge Graph")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Graph statistics
                    st.info(f"Displaying {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} relationships")
                else:
                    st.info("No data matches the current filters. Try adjusting your selection or add more documents.")
                    
            except Exception as e:
                st.error(f"Error rendering graph: {e}")
                st.error("Make sure Neo4j is running and contains data.")
    
    def _render_ai_assistant_tab(self):
        """Render AI assistant chat interface"""
        st.header("💬 AI Knowledge Assistant")
        st.markdown("Ask questions about your knowledge base and get AI-powered answers.")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your knowledge base..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = self.assistant.answer_question(prompt)
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"I encountered an error: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

class PersonalKnowledgeAssistant:
    """Enhanced Personal Knowledge Assistant with Neo4j and Visualization"""
    
    def __init__(self, gemini_api_key: str, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        
        self.processor = DocumentProcessor()
        self.extractor = EnhancedEntityExtractor(gemini_api_key)
        
        # Initialize Neo4j connection
        try:
            self.knowledge_graph = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            raise
        
        # Initialize Gemini client for Q&A
        self.client = genai.Client(api_key=gemini_api_key)
        self.model = "gemini-2.0-flash"
    
    def ingest_document(self, file_path: str) -> bool:
        """Ingest a document into the knowledge graph"""
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Process document
            doc_data = self.processor.process_file(file_path)
            if not doc_data['content'].strip():
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            document_id = doc_data['metadata']['filename']
            
            # Add document to graph
            self.knowledge_graph.add_document(document_id, doc_data['metadata'])
            
            # Extract entities and relationships
            entities, relationships = self.extractor.extract_entities_and_relations(
                doc_data['content'], document_id
            )
            
            # Add entities to graph
            for entity in entities:
                self.knowledge_graph.add_entity(entity)
                self.knowledge_graph.link_entity_to_document(entity.name, document_id)
            
            # Add relationships to graph
            for relationship in relationships:
                self.knowledge_graph.add_relationship(relationship)
            
            logger.info(f"Successfully processed {file_path}: {len(entities)} entities, {len(relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return False
    
    def ingest_directory(self, directory_path: str, file_extensions: List[str] = None):
        """Ingest all supported documents from a directory"""
        if file_extensions is None:
            file_extensions = ['.pdf', '.docx', '.txt', '.md']
        
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return
        
        files_processed = 0
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in file_extensions:
                if self.ingest_document(str(file_path)):
                    files_processed += 1
        
        logger.info(f"Processed {files_processed} files from {directory_path}")
    
    def answer_question(self, question: str, debug: bool = False) -> str:
        """Answer a question using the knowledge graph and LLM"""
        
        # Handle special question types
        question_lower = question.lower().strip()
        
        # Handle company listing questions
        if any(phrase in question_lower for phrase in ['what companies', 'list companies', 'companies mentioned']):
            return self._answer_companies_question()
        
        # Handle "who works where" questions  
        if any(phrase in question_lower for phrase in ['who works where', 'who works at', 'employment', 'work relationships']):
            return self._answer_work_relationships_question()
        
        # Find relevant entities
        relevant_entities = self._find_relevant_entities(question)
        
        if debug:
            print(f"DEBUG: Found {len(relevant_entities)} relevant entities: {relevant_entities}")
        
        if not relevant_entities:
            # Try a broader search if no entities found
            with self.knowledge_graph.driver.session() as session:
                result = session.run("MATCH (e:Entity) RETURN e.name, e.type LIMIT 10")
                all_entities = [(record['e.name'], record['e.type']) for record in result]
                
            if debug:
                print(f"DEBUG: No relevant entities found. Available entities: {all_entities[:5]}")
            
            if all_entities:
                entity_list = ", ".join([f"{name} ({etype})" for name, etype in all_entities[:8]])
                return f"I don't have specific information about that topic, but my knowledge base contains information about: {entity_list}. Feel free to ask about any of these!"
            else:
                return "I don't have information about that topic in my knowledge base. Please add relevant documents first."
        
        # Gather context from knowledge graph
        context = self._gather_context(relevant_entities)
        
        if debug:
            print(f"DEBUG: Context length: {len(context)} characters")
            print(f"DEBUG: Context preview: {context[:200]}...")
        
        # Generate answer using LLM
        answer = self._generate_answer(question, context)
        
        return answer
    
    def _answer_companies_question(self) -> str:
        """Answer questions about companies in the knowledge base"""
        with self.knowledge_graph.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity) 
                WHERE e.type IN ['ORG', 'ORGANIZATION'] 
                RETURN e.name, coalesce(e.description, '') as description
                ORDER BY e.name
            """)
            
            companies = []
            for record in result:
                name = record['e.name']
                desc = record['description']
                if desc:
                    companies.append(f"**{name}**: {desc}")
                else:
                    companies.append(f"**{name}**")
            
            if companies:
                return f"Here are the companies mentioned in my knowledge base:\n\n" + "\n".join(companies)
            else:
                return "I don't have information about any companies in my knowledge base."
    
    def _answer_work_relationships_question(self) -> str:
        """Answer questions about who works where"""
        with self.knowledge_graph.driver.session() as session:
            result = session.run("""
                MATCH (person:Entity)-[r]-(org:Entity)
                WHERE person.type = 'PERSON' AND org.type IN ['ORG', 'ORGANIZATION']
                AND (type(r) CONTAINS 'WORK' OR type(r) = 'WORKS_AT' OR type(r) = 'CEO' OR type(r) = 'FOUNDER')
                RETURN person.name as person, type(r) as relationship, org.name as organization
                ORDER BY person.name
            """)
            
            relationships = []
            for record in result:
                person = record['person']
                rel = record['relationship']
                org = record['organization']
                relationships.append(f"• **{person}** {rel.lower().replace('_', ' ')} **{org}**")
            
            if relationships:
                return f"Here are the work relationships I found:\n\n" + "\n".join(relationships)
            else:
                return "I don't have specific information about who works where in my knowledge base."
    
    def _find_relevant_entities(self, question: str) -> List[str]:
        """Find entities relevant to the question with improved logic"""
        question_lower = question.lower()
        
        # Extract key terms and names from question
        key_terms = []
        
        # Look for proper nouns (capitalized words)
        import re
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', question)
        key_terms.extend(proper_nouns)
        
        # Add the original question for broad search
        key_terms.append(question)
        
        # Split question into individual words for broader search
        words = [word.strip('.,!?').lower() for word in question.split() if len(word) > 2]
        key_terms.extend(words)
        
        # Search with each term
        all_results = []
        seen_entities = set()
        
        with self.knowledge_graph.driver.session() as session:
            for term in key_terms:
                if term and len(term) > 1:
                    # Direct search
                    search_results = self.knowledge_graph.search_entities(term, limit=15)
                    for result in search_results:
                        entity_name = result['name']
                        if entity_name not in seen_entities:
                            seen_entities.add(entity_name)
                            all_results.append(entity_name)
            
            # If still no results, try different strategies based on question type
            if not all_results:
                if any(word in question_lower for word in ['who', 'people', 'person', 'works', 'founder']):
                    # Look for people
                    result = session.run("MATCH (e:Entity) WHERE e.type = 'PERSON' RETURN e.name LIMIT 10")
                    all_results.extend([record['e.name'] for record in result])
                
                elif any(word in question_lower for word in ['company', 'companies', 'organization', 'org']):
                    # Look for organizations
                    result = session.run("MATCH (e:Entity) WHERE e.type IN ['ORG', 'ORGANIZATION'] RETURN e.name LIMIT 15")
                    all_results.extend([record['e.name'] for record in result])
                
                elif any(word in question_lower for word in ['technology', 'tech', 'software', 'ai', 'ml']):
                    # Look for concepts and technologies
                    result = session.run("MATCH (e:Entity) WHERE e.type IN ['CONCEPT', 'TECHNOLOGY'] RETURN e.name LIMIT 10")
                    all_results.extend([record['e.name'] for record in result])
                
                else:
                    # Fallback: get some entities based on centrality
                    result = session.run("""
                        MATCH (e:Entity)
                        OPTIONAL MATCH (e)-[r]-()
                        WITH e, count(r) as connections
                        ORDER BY connections DESC
                        LIMIT 10
                        RETURN e.name
                    """)
                    all_results.extend([record['e.name'] for record in result])
        
        return all_results[:15]  # Return top 15 relevant entities
    
    def _gather_context(self, entity_names: List[str]) -> str:
        """Gather context information for entities"""
        context_parts = []
        
        for entity_name in entity_names[:8]:  # Increased limit for more context
            details = self.knowledge_graph.get_entity_details(entity_name)
            
            if details:
                context_parts.append(f"**Entity: {entity_name}**")
                context_parts.append(f"Type: {details['type']}")
                
                if details['description']:
                    context_parts.append(f"Description: {details['description']}")
                
                # Add relationships
                if details['outgoing_relations']:
                    context_parts.append("Outgoing Relationships:")
                    for rel in details['outgoing_relations'][:5]:  # Limit relationships
                        if rel['target']:
                            context_parts.append(f"  - {entity_name} {rel['relation']} {rel['target']}")
                            if rel.get('description'):
                                context_parts.append(f"    ({rel['description']})")
                
                if details['incoming_relations']:
                    context_parts.append("Incoming Relationships:")
                    for rel in details['incoming_relations'][:3]:
                        if rel['source']:
                            context_parts.append(f"  - {rel['source']} {rel['relation']} {entity_name}")
                
                if details['source_documents']:
                    context_parts.append(f"Source Documents: {', '.join(details['source_documents'])}")
                
                context_parts.append("")  # Separator
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with knowledge graph context"""
        
        prompt = f"""
        You are a helpful AI assistant with access to a personal knowledge base. Answer the user's question based ONLY on the provided context from their knowledge graph.

        Context from Knowledge Graph:
        {context}

        User Question: {question}

        Instructions:
        1. Provide a comprehensive answer based ONLY on the available information in the context above
        2. Reference specific entities and relationships when relevant
        3. If the context contains relevant information, use it to answer the question fully
        4. Be conversational and helpful
        5. If you can make connections between different pieces of information, do so
        6. For questions about "who works where" or relationships, list all relevant connections you can find
        7. For questions about companies or people, provide all relevant details from the context
        8. If the context doesn't contain enough information for a complete answer, say what information IS available

        Important: Base your answer ENTIRELY on the context provided above. Do not use external knowledge.

        Answer:
        """
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )
            return response.text
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        return self.knowledge_graph.get_graph_statistics()
    
    def search_knowledge_base(self, query: str) -> List[Dict]:
        """Search the knowledge base for entities and relationships"""
        search_results = self.knowledge_graph.search_entities(query)
        
        results = []
        for result in search_results:
            results.append({
                'entity_name': result['name'],
                'entity_type': result['type'],
                'description': result.get('description', ''),
                'source_documents': []  # Could be populated from entity details if needed
            })
        
        return results
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'knowledge_graph'):
            self.knowledge_graph.close()

# Main application
def main():
    """Main application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Personal Knowledge Assistant with Neo4j")
    parser.add_argument("--gemini-api-key", required=True, help="Gemini API key")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--ingest", help="Ingest file or directory")
    parser.add_argument("--question", help="Ask a question")
    parser.add_argument("--clear-db", action="store_true", help="Clear database (use with caution!)")
    
    args = parser.parse_args()
    
    try:
        # Initialize assistant
        assistant = PersonalKnowledgeAssistant(
            gemini_api_key=args.gemini_api_key,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password
        )
        
        if args.clear_db:
            assistant.knowledge_graph.clear_database()
            print("Database cleared!")
            return
        
        if args.web:
            # Launch Streamlit interface
            interface = StreamlitInterface(assistant)
            interface.render()
        
        elif args.ingest:
            path = Path(args.ingest)
            if path.is_file():
                assistant.ingest_document(str(path))
            elif path.is_dir():
                assistant.ingest_directory(str(path))
            else:
                print(f"Path not found: {args.ingest}")
        
        elif args.question:
            answer = assistant.answer_question(args.question)
            print(f"\nQuestion: {args.question}")
            print(f"Answer: {answer}")
        
        else:
            print("Personal Knowledge Assistant")
            print("Use --web for web interface or --help for options")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if 'assistant' in locals():
            assistant.close()

if __name__ == "__main__":

    main()


