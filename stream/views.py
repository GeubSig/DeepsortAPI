import datetime

from django.core import serializers
from django.db.models import Sum
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
# from rest_framework.response import Response

from stream.streaming import stream, people_counter
from stream.models import InOut, InOutTimestamp, InoutTimeline


def index(request):
    return render(request, 'index.html')


def video_feed1(request):
    source = 'videos/in.mp4'
    return StreamingHttpResponse(stream(source, is_in=True), content_type='multipart/x-mixed-replace; boundary=frame')


def video_feed2(request):
    # source = 'rtsp://localhost:8554/test'
    source = 'videos/CH02-out.mp4'

    return StreamingHttpResponse(stream(source, is_in=False), content_type='multipart/x-mixed-replace; boundary=frame')


def days_data(request):
    breakfast_inout = InOut.objects.filter(meal_type=0).order_by("-date")[:7]  # .values_list("date", flat=True)
    breakfast_dates = [ob.get_date() for ob in breakfast_inout]
    breakfast_people = [ob.get_total_people() for ob in breakfast_inout]

    lunch_inout = InOut.objects.filter(meal_type=1).order_by("-date")[:7]
    lunch_dates = [ob.get_date() for ob in lunch_inout]
    lunch_people = [ob.get_total_people() for ob in lunch_inout]

    dinner_inout = InOut.objects.filter(meal_type=2).order_by("-date")[:7]
    dinner_dates = [ob.get_date() for ob in dinner_inout]
    dinner_people = [ob.get_total_people() for ob in dinner_inout]

    return JsonResponse({
        "breakfastDate": breakfast_dates,
        "breakfastPeople": breakfast_people,
        "lunchDate": lunch_dates,
        "lunchPeople": lunch_people,
        "dinnerDate": dinner_dates,
        "dinnerPeople": dinner_people,
    })


def times_data(request):
    breakfast_in_people_sum = []
    lunch_in_people_sum = []
    dinner_in_people_sum = []

    for i in range(5, 61, 5):
        # 값이 없으면 0
        breakfast_val = InOutTimestamp.objects.filter(five_minutes=i, meal_type=0).aggregate(Sum('people'))["people__sum"]
        breakfast_in_people_sum.append(breakfast_val if breakfast_val is not None else 0)
    for i in range(5, 61, 5):
        lunch_val = InOutTimestamp.objects.filter(five_minutes=i, meal_type=1).aggregate(Sum('people'))["people__sum"]
        lunch_in_people_sum.append(lunch_val if lunch_val is not None else 0)
    for i in range(5, 61, 5):
        dinner_val = InOutTimestamp.objects.filter(five_minutes=i, meal_type=2).aggregate(Sum('people'))["people__sum"]
        dinner_in_people_sum.append(dinner_val if dinner_val is not None else 0)

    return JsonResponse({"breakfast": breakfast_in_people_sum, "lunch": lunch_in_people_sum, "dinner": dinner_in_people_sum})


def current_people(request):
    waiting_time = 0 # Todo 대기시간 예측 알고리즘 짜야함
    return JsonResponse({"currentPeople": people_counter.count, "waitingTime": waiting_time})


def timeline(request):
    datas = InoutTimeline.objects.order_by("-id")[:24]
    times = [ob.get_hour() for ob in datas]
    people = [ob.get_people() for ob in datas]
    times.reverse()
    people.reverse()

    return JsonResponse({"times": times, "people": people})

# def add_dummy(request):
#     import datetime
#     import random

#     print('req')
#
#     # 오늘 날짜
#     base = datetime.datetime.today()
#     # 오늘 날짜부터 7일전 까지의 날짜 리스트
#     date_list = [base - datetime.timedelta(days=x) for x in range(7)]
#
#     for date in date_list:
#         for meal_type in range(3):
#             total_people = random.randrange(180, 220)
#
#             inout = InOut()
#             inout.date = date
#             inout.meal_type = meal_type
#             inout.total_people_in = total_people
#             inout.save()
#
#             for five_min_term in range(5, 61, 5):
#                 people = random.randrange(0, 90)
#
#                 inout_timestamp = InOutTimestamp()
#                 inout_timestamp.five_minutes = five_min_term
#                 inout_timestamp.meal_type = meal_type
#                 inout_timestamp.people = people
#                 inout_timestamp.inout_key = inout
#                 inout_timestamp.save()
#
#     for hour in range(0, 24):
#         people = random.randrange(0, 90)
#
#         inout_timeline = InoutTimeline()
#         inout_timeline.people = people
#         inout_timeline.hour = hour
#         inout_timeline.save()
#
#     return JsonResponse({"hi": "hi"})
