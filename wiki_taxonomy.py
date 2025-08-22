import streamlit as st
import requests
from PIL import Image
from transformers import pipeline
from bs4 import BeautifulSoup

@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="microsoft/resnet-50")

def predict_species(image):
    classifier = load_classifier()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    predictions = classifier(image)
    predicted_species = predictions[0]['label']
    confidence = predictions[0]['score']
    return predicted_species, confidence

@st.cache_data
def get_wikipedia_title(name):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={name}&format=json"
        res = requests.get(url, timeout=10)
        data = res.json()
        if data["query"]["search"]:
            return data["query"]["search"][0]["title"]
        return name
    except Exception as e:
        st.warning(f"Wikipedia search error: {e}")
        return name


# @st.cache_data
from functools import lru_cache

@st.cache_data
def get_taxonomy_from_wikidata(label):
    try:
        # Step 1: Search for label on Wikidata
        search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={label}&language=en&format=json"
        search_response = requests.get(search_url, timeout=5).json()
        if not search_response["search"]:
            return {"error": f"No Wikidata entity found for label '{label}'."}

        entity_id = search_response["search"][0]["id"]

        # Step 2: Try to trace to valid taxon
        @lru_cache(maxsize=128)
        def fetch_entity_data(eid):
            url = f"https://www.wikidata.org/wiki/Special:EntityData/{eid}.json"
            return requests.get(url, timeout=5).json()["entities"][eid]

        def find_taxon_entity(eid, depth=0):
            if depth > 5:
                return None
            ent = fetch_entity_data(eid)
            claims = ent.get("claims", {})
            if "P225" in claims and "P171" in claims:
                return eid
            for prop in ["P31", "P279"]:
                for val in claims.get(prop, []):
                    try:
                        next_id = val["mainsnak"]["datavalue"]["value"]["id"]
                        found = find_taxon_entity(next_id, depth+1)
                        if found:
                            return found
                    except:
                        continue
            return None

        taxon_entity = find_taxon_entity(entity_id)
        if not taxon_entity:
            return {"error": f"Could not find taxonomic root from entity '{entity_id}'."}

        taxonomy = {}

        def fetch_taxonomy(eid, level=0):
            if level > 20:
                return
            ent = fetch_entity_data(eid)
            claims = ent.get("claims", {})

            sci_name = claims.get("P225", [{}])[0].get("mainsnak", {}).get("datavalue", {}).get("value", "")
            rank_id = claims.get("P105", [{}])[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id", "")

            rank = ""
            if rank_id:
                try:
                    rank_ent = fetch_entity_data(rank_id)
                    rank = rank_ent["labels"].get("en", {}).get("value", "")
                except:
                    pass

            if rank and sci_name:
                taxonomy[rank.capitalize()] = sci_name
            elif sci_name:
                taxonomy[f"Unranked {level}"] = sci_name

            parent_id = claims.get("P171", [{}])[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id", "")
            if parent_id:
                fetch_taxonomy(parent_id, level+1)

        fetch_taxonomy(taxon_entity)

        ordered = {}
        for key in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species", "Subspecies"]:
            if key in taxonomy:
                ordered[key] = taxonomy[key]

        return ordered if ordered else {"error": f"No taxonomy could be extracted from entity '{taxon_entity}'."}
    except Exception as e:
        return {"error": f"Wikidata taxonomy error: {e}"}


def display_taxonomy(taxonomy):
    st.subheader("🧬 Taxonomic Classification (from Wikidata)")
    if "error" in taxonomy:
        st.warning(taxonomy["error"])
        return

    for rank in ["Kingdom", "Phylum", "Class", "Order", "Family", "Subfamily", "Genus", "Species", "Subspecies"]:
        if rank in taxonomy:
            st.markdown(f"**{rank}:** `{taxonomy[rank]}`")

def main():
    # st.set_page_config(page_title="Animal Taxonomy Classifier (Wikipedia)", layout="wide")
    st.title("🧬 Animal Taxonomy Classifier")

    uploaded_image = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing and fetching taxonomy..."):
                try:
                    predicted_species, confidence = predict_species(image)
                    st.success(f"🎯 Predicted: **{predicted_species}** ({confidence:.2%} confidence)")

                    corrected_title = get_wikipedia_title(predicted_species)
                    st.info(f"🔍 Wikipedia Title: **{corrected_title}**")

                    taxonomy = get_taxonomy_from_wikidata(corrected_title)
                    display_taxonomy(taxonomy)

                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
