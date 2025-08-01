import streamlit as st

from search_module import link_scrapper_perplexity

product_url = st.text_input("Enter link")
if product_url:
    description = link_scrapper_perplexity(product_url)
    st.json(description)

