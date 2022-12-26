from django.db import models


class InOut(models.Model):
    # id = models.AutoField()
    date = models.DateField()
    meal_type = models.IntegerField()
    meal_start_time = models.DateTimeField()
    total_in_people = models.IntegerField()
    created_at = models.DateTimeField()


class InOutTimestamp(models.Model):
    # id = models.AutoField()
    time = models.DateTimeField()
    people_in = models.IntegerField()
    people_out = models.IntegerField()
    inout_key = models.ForeignKey(to=InOut, related_name='inout_timestamp', on_delete=models.CASCADE, db_column='inout_key')
