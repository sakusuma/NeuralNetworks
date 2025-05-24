# Databricks notebook source
a = []

# COMMAND ----------

a.append(1)

# COMMAND ----------

print(a)

# COMMAND ----------

a.append([1,2,3])

# COMMAND ----------

print(a)

# COMMAND ----------

a.extend([2,3,4])

# COMMAND ----------

print(a)

# COMMAND ----------

a.pop()

# COMMAND ----------

a

# COMMAND ----------

a.pop(2)

# COMMAND ----------

a

# COMMAND ----------

mail = 'Hi Sai, how are you?'

# COMMAND ----------

mail.split()

# COMMAND ----------

mail.split(',')

# COMMAND ----------

list_mail = mail.split()

# COMMAND ----------

list_mail

# COMMAND ----------

' '.join(list_mail)

# COMMAND ----------

list = ['Hi']

# COMMAND ----------

print(list*5)

# COMMAND ----------

print(list+list)

# COMMAND ----------

len(list_mail)

# COMMAND ----------

sorted(list_mail)

# COMMAND ----------

5%2

# COMMAND ----------

first_tuple = (1,2,3)

# COMMAND ----------

first_tuple

# COMMAND ----------

first_tuple[0]

# COMMAND ----------

tup = (1, 2, 3, 4)

# COMMAND ----------

a = [i for i in tup]

# COMMAND ----------

a

# COMMAND ----------

dict_1 = {}

# COMMAND ----------

dict_1

# COMMAND ----------

dict_2 = {
    'a':1,
    'b':2
}

# COMMAND ----------

print(dict_2)

# COMMAND ----------

d = dict(a=1, b=2)

# COMMAND ----------

d

# COMMAND ----------

# appending to dict
d['c'] = 3


# COMMAND ----------

d

# COMMAND ----------

d['a']

# COMMAND ----------

d.get('f', 'NA')

# COMMAND ----------

d['d'] = [1,2,3]

# COMMAND ----------

d

# COMMAND ----------

'd' in d

# COMMAND ----------

print('d' in d)

# COMMAND ----------

d.keys()

# COMMAND ----------

d.values()

# COMMAND ----------

for key, value in d.items():
    print(key, ':', value)

# COMMAND ----------

del d['d']

# COMMAND ----------

del d['d']

# COMMAND ----------

d

# COMMAND ----------

list_mail = ['F','A','A','N','G']

# COMMAND ----------

set1 = set(list_mail)
set1

# COMMAND ----------

set2 = {'A', 'A', 'B', 'C'}

# COMMAND ----------

set2

# COMMAND ----------

set1.difference(set2)

# COMMAND ----------

set1.intersection(set2)

# COMMAND ----------

set1.union(set2)

# COMMAND ----------

print('The alphabet set is {0}'.format(set1))

# COMMAND ----------

