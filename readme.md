1. Create a .env file.
2. Set up the following env variables:
   1. GOOGLE_API_KEY
   2. OXYLABS_USERNAME 
   3. OXYLABS_PASSWORD
   4. FASHNAPI
3. Install all the dependencies from requirements.txt
4. Run main.py using uvicorn main:app --reload --port <port_number>
5. For testing the measurements endpoint 
   1. Make sure you have the height of the person who's photo is uploaded.
   2. For front photo make sure the person is standing upright against a plain surface with a slight distance between his hands and torso.
   3. For side photo make sure the person is standing upright with the photo being taken perpendicularly to him.
   4. Avoid flowy dresses like skirts or gowns, go for tighter clothing to get the best results.