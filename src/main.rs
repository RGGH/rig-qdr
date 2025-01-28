#![allow(unused)]
use dotenv::dotenv;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, QueryPointsBuilder, UpsertPointsBuilder,
        Value, VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use std::collections::HashMap;

use rig::{providers::openai::Client, vector_store::VectorStoreIndex};
use rig_qdrant::QdrantVectorStore;
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

const COLLECTION_NAME: &str = "rig-collection";

// Shape of data that needs to be RAG'ed.
// The definition field will be used to generate embeddings.
#[derive(Clone, Deserialize, Debug, Serialize, Eq, PartialEq, Default)]
struct SimpleDocument<'a>(&'a str);

impl<'a> AsRef<str> for SimpleDocument<'a> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    dotenv().ok();
    let openai_client = Client::from_env();

    let qdrant_client = Qdrant::from_url("http://localhost:6334").build()?;

    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
    )?;

    let documents = vec![
        SimpleDocument("Some text"),
        SimpleDocument("Mofo "),
        SimpleDocument("wwW"),
    ];

    // Generate embeddings
    let embeddings = model.embed(documents.clone(), None)?;
    println!("\n\n✅ Embeddings length: {}", embeddings.len());

    // Ensure embeddings and documents are aligned
    if embeddings.len() != documents.len() {
        return Err(anyhow::anyhow!(
            "Embedding length does not match document count"
        ));
    }


    // Map documents and embeddings to Qdrant points
    let points: Vec<PointStruct> = documents
        .into_iter()
        .zip(&embeddings)
        .map(|(doc, embedding)| {
            //
            let mut payload = HashMap::new();
            payload.insert("document".to_string(), Value::from(doc.0)); // Use Value::from

            PointStruct::new(
                Uuid::new_v4().to_string(), // Unique point ID
                embedding.clone(),          // The embedding vector for the document
                payload,
            )
        })
        .collect();

    // Check if the collection exists, create it if it doesn't
    if !qdrant_client.collection_exists(COLLECTION_NAME).await? {
        qdrant_client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(384 as u64, Distance::Cosine)),
            )
            .await?;
    } else {
        println!("Collection `{}` already exists!", COLLECTION_NAME);
    }

    qdrant_client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points))
        .await?;

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);

    // ------- query
    let query_vector = embeddings[0].clone();

    // Perform a query in Qdrant

    let search_result = qdrant_client
    .query(
        QueryPointsBuilder::new(COLLECTION_NAME)
            .query(query_vector)
    )
    .await?;

dbg!(search_result);

    Ok(())
}
