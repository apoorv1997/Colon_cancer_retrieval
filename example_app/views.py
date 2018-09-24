import json
from django.views.generic.base import TemplateView
from django.views.generic import View
from django.http import JsonResponse
from chatterbot import ChatBot
from chatterbot.ext.django_chatterbot import settings
from example_app.net import SiameseNetwork
import torch
import pickle
import cv2

saved_model=torch.load('example_app/model-epoch-60.pth',map_location=lambda storage, loc: storage)
model = SiameseNetwork()
model.load_state_dict(saved_model)
indexed_data = pickle.load(open("example_app/preprocessing.pickle","rb"))

class MainWebsite(TemplateView):
    template_name = 'index.html'

class ChatterBotAppView(TemplateView):
    template_name = 'app.html'

class MLView(View):
    def post(request, *args, **kwargs):
        print(request,request.keys()``)
        # j=request["POST"]['query_image'].save("image.jpg")
        print("YAY",j)
        print(request.FILES['file'])
        # t = cv2.imread(request.FILES['file'])
        x = Variable(x, volatile=True)
        output = model.forward_once(x).detach().cpu().numpy()
        distance = np.sqrt(np.sum(np.power(train_vectors-output,2),axis=1))
        # print(distance.shape,train_vectors.shape)
        # print()
        mink = np.argsort(distance)[:k]
        # print("MINDITANCE:{}".format(distance[mink[0]]))
        # print("DISTANCE:{}",distance)
        retrieved = np.flatnonzero(distance < 0.3)
        return JsonResponse("HELLO", status=200)


class ChatterBotApiView(View):
    """
    Provide an API endpoint to interact with ChatterBot.
    """

    chatterbot = ChatBot(**settings.CHATTERBOT)

    def post(self, request, *args, **kwargs):
        """
        Return a response to the statement in the posted data.

        * The JSON data should contain a 'text' attribute.
        """
        input_data = json.loads(request.read().decode('utf-8'))

        if 'text' not in input_data:
            return JsonResponse({
                'text': [
                    'The attribute "text" is required.'
                ]
            }, status=400)

        response = self.chatterbot.get_response(input_data)

        response_data = response.serialize()

        return JsonResponse(response_data, status=200)

    def get(self, request, *args, **kwargs):
        """
        Return data corresponding to the current conversation.
        """
        return JsonResponse({
            'name': self.chatterbot.name
        })
