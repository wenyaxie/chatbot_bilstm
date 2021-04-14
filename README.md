# Instructions

Skip to step 5 if the model has been trained.

1. Create a venv by `python3 -m venv .venv`
2. Activate venv by `source .venv/bin/activate`
3. Install dependencies by `pip install -r requirements.txt`
4. Train model by `python3 main.py train`
5. Run server
    - Run server without Docker:
        1. Start server by `python3 main.py serve`
        2. Send request, e.g. `curl -X POST --data '{"question": "What kinds of AI papliatcions can be etsted by tihs tool? Can I test NLP paplicaitosn?"}' -H "Content-Type: application/json"  localhost:22370`
    - Run server with Docker:
        1. Build image by `docker build . -t bilstm:latest`
        2. Run the image in a container by `docker run -p 22370:22370 bilstm`
        3. Send request, e.g. `curl -X POST --data '{"question": "What kinds of AI papliatcions can be etsted by tihs tool? Can I test NLP paplicaitosn?"}' -H "Content-Type: application/json"  localhost:22370`
        4. At the end, kill the container by `docker rm $(docker stop $(docker ps -a -q --filter ancestor=bilstm --format="{{.ID}}"))`
        5. Clean up the image by `docker image rm bilstm -f`
6. Push image to Docker Hub by `docker tag bilstm:latest wenyaxie/bilstmchatbot && docker push wenyaxie/bilstmchatbot`
