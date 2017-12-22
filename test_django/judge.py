# -*- coding: utf-8 -*-

from django.http import HttpResponse
from django.shortcuts import render_to_response
import pandas as pd
from message_classcifier import Message_Classcifier
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render

pos, neg = 0, 1
clf = Message_Classcifier()
clf.load_model('final_model')


def index(request):
    context = {}
    return render(request, 'index.html', context)


# 接收请求数据
@csrf_exempt
def search(request):
    query = request.POST['message']
    print query
    result = clf.predict(query)[0]
    request.encoding = 'utf-8'
    message = '垃圾短信' if result == neg else '正常短信'
    print message
    return JsonResponse({'message':message})