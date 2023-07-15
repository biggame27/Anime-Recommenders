import streamlit as st
import anime_recommendations as ar

# def load_model():
#     with open('anime.pkl', 'rb') as file:
#         data = pickle.load(file)
#     return datar

def show_page():
    st.title("Anime Recommender")
    st.markdown("![Alt Text](https://wallpaperaccess.com/thumb/17350.jpg)")
    st.write("Note: There isn't a search engine so the anime has to exactly match the dataset:(.")
    st.write("Here are some recommendations:")
    arr = ['Gintama', 'Tokyo Ghoul', 'Fullmetal Alchemist: Brotherhood', 'Haikyuu!!', 'No Game No Life', 'Naruto', 'Sword Art Online', 'Shingeki no Kyojin']
    for i in arr:
        st.text(f"- {i}")
    user_input = st.text_input(f"**Put the anime here for a recommendation!**", 'Dragon Ball Z')
    
    ok = st.button("Find Recommendation")

    if ok:
        anime = ar.anime_recommendation(user_input)
        if anime == []:
            st.write("Not found ;c")
        else:
            for i in anime:
                st.write(i)
show_page()