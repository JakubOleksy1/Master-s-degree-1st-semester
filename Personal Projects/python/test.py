import os

os.chdir("C:/Users/jakub/Visual Studio Code/Personal Projects/python")

#"""
#110. Funkcje generujące - słowo kluczowe yield
def generate_inf_numbers():
    number = 0
    while True:
        number = number + 1
        yield number*number

generatedNumbers = []

numberGenerated = generate_inf_numbers()
"""
for _ in range(20):
    generatedNumbers.append(next(numberGenerated))

print(generatedNumbers)

for _ in range(30):
    generatedNumbers.append(next(numberGenerated))

print(generatedNumbers)
"""
#"""






"""
#110. Funkcje generujące - słowo kluczowe yield
def generate_even_numbers():
    print("start")
    for element in range(400):
        print("przed yield")
        if(element % 2 == 0):
            yield element
            print("po yeild")

evenNumbersGenerator = (element
                        for element in range(400)
                        if (element % 2 == 0)
                        )

a = generate_even_numbers()

def generate_10_numbers():
    x = 0
    while x < 10:
        yield x
        x = x + 1

print(list(generate_10_numbers()))

generate_10_numbers_expression = (x
                                  for x in range(10))

print(list(generate_10_numbers_expression))
print(list(generate_10_numbers_expression))
"""







"""
#106. ĆWICZENIE: System zarządzania ulubionymi kotkami - wstęp do ćwiczenia
#107. Dodawanie kotow do ulubionych'
#108. Usuwanie kotow z ulubionych
#109. Optymalizacja i refaktoryzacja kodu z poprzednich lekcji
import requests
import json
import webbrowser
import credentials

from pprint import pprint

def print_favourite_cats(favouriteCats):
    print("\nTwoje ulubione kotki to:")
    for cat in favouriteCats:
        print(cat["id"], cat['image']['url'])

def print_random_cat(randomCat):
    print("\nWylosowano kotka: ", randomCat["url"])

def decision_add(addToFavourites, newlyAddedCatInfo):
    if(addToFavourites.upper() == "T"):
        resultFromAddingFavouriteCat = add_favourite_cat(randomCat["id"], userId)
        newlyAddedCatInfo = {resultFromAddingFavouriteCat["id"] : randomCat["url"]}
    else:
        print("\nNie dodano kota ")
    return newlyAddedCatInfo

def decision_delete(deleteCat):
    if(deleteCat.upper() == "T"):
        favouriteCatId = input("Podaj id: ")
        print(remove_favourite_cat(userId, favouriteCatId))
    else:
        print("\nNie usuwam kota ")

def get_json_content_from_response(response):
    try:
        content = response.json()
    except json.decoder.JSONDecodeError:
        print("Niepoprawny format", response.text)
    else:
        return content

def get_favourite_cats(userId):
    params = {
        "sub_id" : userId
        }
    r = requests.get('https://api.thecatapi.com/v1/favourites/', params,
                 headers=credentials.headers)
    
    return get_json_content_from_response(r)

def get_random_cat():
    r = requests.get('https://api.thecatapi.com/v1/images/search',
                 headers=credentials.headers)
    
    return get_json_content_from_response(r)[0]

def add_favourite_cat(catId, userId):
    catData = {
        "image_id" : catId,
        "sub_id" : userId
        }
    r = requests.post('https://api.thecatapi.com/v1/favourites/', json = catData,
                 headers=credentials.headers)
    
    return get_json_content_from_response(r)

def remove_favourite_cat(userId, favouriteCatId):
    r = requests.delete('https://api.thecatapi.com/v1/favourites/'+favouriteCatId,                headers=credentials.headers)
    
    return get_json_content_from_response(r)

if __name__ == "__main__":
    print("Podaj login i haslo\n")

    userId = "agh2m"
    name = "Jakub"

    print("Witaj " + name)

    favouriteCats = get_favourite_cats(userId)

    print_favourite_cats(favouriteCats)

    randomCat = get_random_cat()

    print_random_cat(randomCat)

    addToFavourites = input("Czy chcesz go dodac do ulubionych? T/N\n")

    newlyAddedCatInfo = {}
    newlyAddedCatInfo = decision_add(addToFavourites, newlyAddedCatInfo)

    favouriteCatsById = {
        favouriteCat["id"] : favouriteCat["image"]["url"]
        for favouriteCat in favouriteCats
    }

    favouriteCatsById.update(newlyAddedCatInfo)

    print(favouriteCatsById)

    deleteCat = input("Czy chcesz usunac kota z ulubionych? T/N\n")

    decision_delete(deleteCat)
"""






"""
#103. ĆWICZENIE: Losowe zdjęcia kotów wybranej rasy
import requests
import json
import webbrowser
from pprint import pprint
import random

def pobierz_losowe_koty_dla_rasy(rasa, limit_kotow):
    url = "https://api.thecatapi.com/v1/images/search"
    params = {
        "mime_types": "jpg,png",
        "limit": limit_kotow,
        "breed_ids": rasa
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        koty = r.json()
        losowe_koty = random.sample(koty, min(limit_kotow, len(koty)))  # Losowy wybór trzech kotów spośród wszystkich pobranych
        for index, kot in enumerate(losowe_koty, start=1):
            print(f"Kot {index}:")
            print(f"ID: {kot['id']}")
            webbrowser.open_new_tab(kot['url'])  # Otwieranie nowej karty przeglądarki z URL obrazu
            print()
    else:
        print("Wystąpił problem z pobraniem danych.")

def pobierz_dostepne_rasy():
    url = "https://api.thecatapi.com/v1/breeds"
    r = requests.get(url)
    if r.status_code == 200:
        rasy = r.json()
        print("Dostępne rasy kotów:")
        for rasa in rasy:
            print(rasa["name"])
    else:
        print("Wystąpił problem z pobraniem danych.")

pobierz_dostepne_rasy()
rasa = input("\nWybierz rasę kota: ")
limit_kotow = int(input("\nIle kota: "))
pobierz_losowe_koty_dla_rasy(rasa, limit_kotow)
"""





"""
#102. ĆWICZENIE: Fakty o kotach
import requests
import json
import webbrowser
from pprint import pprint

params = {
   "amount" : 5,
}
jpeg = {
}
u = 1
while(u != 3):
    u = int(input("\n\nWhat animal would you like to see facts about?\n1.Cats\n2.Dogs\n3.End program\n"))
    if(u == 1):
        params["animal_type"] = "cat"
        #jpeg["link"] = "https://cataas.com/cat"
    elif(u == 2):
        params["animal_type"] = "dog"
        #jpeg["link"] = "https://dog.ceo/api/breeds/image/random"
    elif(u == 3):
        pprint("Naura")
        break
    else:
        print("\nYou wrote wrong value I will ask again\n")
        continue
            
    r = requests.get("https://cat-fact.herokuapp.com/facts/random", params)

    try:
        content = r.json()
    except json.decoder.JSONDecodeError:
        print("Niepoprawny format")
    else:
        for animal in content:
           print(animal["text"])

    #webbrowser.open_new_tab(jpeg["link"])


"""






"""
#98. Czym jest publiczne API? stackoverflow.com API
#99. Pobieranie JSON ze stackoverflow.com
import requests
import json
import pprint
from datetime import datetime, timedelta
import webbrowser

params = {
   "site" : "stackoverflow" ,
   "sort" : "votes",
   "order" : "desc",
   "tagged" : "python",
   "min" : 5
}

current_date = datetime.now()
from_date = current_date - timedelta(days=3)
from_date_str = from_date.strftime('%Y-%m-%d') #sposob 1 

params["fromdate"] = from_date_str
'''
params = {
   "site" : "stackoverflow" ,
   "sort" : "votes",
   "order" : "desc",
  # "fromdate" : int(from_date.timestamp()),
   "tagged" : "python",
   "min" : 5
}
'''
r = requests.get("https://api.stackexchange.com/2.2/questions", params)

try:
    questions = r.json()
except json.decoder.JSONDecodeError:
    print("Niepoprawny format")
else:
    for question in questions["items"]:
        webbrowser.open_new_tab(question["link"])

"""






"""
#ĆWICZENIE: Wczytanie szerokości/wysokości obrazka - biblioteka PILLOW

from PIL import Image

fileName = "maxresdefault.jpg"

image = Image.open(fileName)

width, height = image.size

print("Szerokosc ", width)
print("Wysokosc ", height)
"""






"""
#92. ĆWICZENIE: Przetwarzamy JSON pobrany z poprzedniej lekcji - wręczamy ciasteczko!
#93. SPOSÓB 1: Pobranie jednocześnie kilku wybranych użytkowników z serwera
#94. SPOSÓB 2: Pobranie jednocześnie kilku wybranych użytkowników z serwera
#95. SPOSÓB 3: Pobranie jednocześnie kilku wybranych użytkowników z serwera
import requests
import json
from collections import defaultdict

def count_task_frequency(tasks):
    completedTasksFrequencyByUser = defaultdict(int)
    for entry in tasks:
        if(entry["completed"] == True):
            try:
                completedTasksFrequencyByUser[entry["userId"]] += 1
            except KeyError:
                completedTasksFrequencyByUser[entry["userId"]] = 1

    return completedTasksFrequencyByUser

def get_keys_with_top_values(my_dict):
    return [
        key
        for (key, value) in my_dict.items()
        if value == max(my_dict.values())
    ]

def get_users_with_top_completed_tasks(completedTasksFrequencyByUser):
    userIdWithMaxCompletedAmountOfTasks = []
    maxAmountOfCompletedTask = max(completedTasksFrequencyByUser.values())
    for userId, numberOfCompletedTask in completedTasksFrequencyByUser.items():
        if(numberOfCompletedTask == maxAmountOfCompletedTask):
            userIdWithMaxCompletedAmountOfTasks.append(userId)

    return userIdWithMaxCompletedAmountOfTasks 

def find_name(users, userIdWithMaxCompletedAmountOfTasks):
    for user in users:
        if (user["id"] in userIdWithMaxCompletedAmountOfTasks):
            print("Wreczamy ciastko do uzytkownika o imieniu ", user["name"])
            userIdWithMaxCompletedAmountOfTasks.remove(user["id"])

r = requests.get("https://jsonplaceholder.typicode.com/todos")

try:
    tasks = r.json()
except json.decoder.JSONDecodeError:
    print("Niepoprawny format")
else:
    completedTaskFrequencyByUser = count_task_frequency(tasks)
    userIdWithMaxCompletedAmountOfTasks = get_users_with_top_completed_tasks(completedTaskFrequencyByUser)
    print("Wreczamy ciasteczko mistrza dyscypliny do uzytkownikow o id:", userIdWithMaxCompletedAmountOfTasks)

#sposob 1
r = requests.get("https://jsonplaceholder.typicode.com/users")
try:
    users = r.json()
except json.decoder.JSONDecodeError:
    print("Zly format")
else:
    find_name(users, userIdWithMaxCompletedAmountOfTasks)

#sposob 2
for userId in userIdWithMaxCompletedAmountOfTasks:
    #r = requests.get("https://jsonplaceholder.typicode.com/users/" + str(userId))
    r = requests.get("https://jsonplaceholder.typicode.com/users/", params="id="+ str(userId))
    
    user = r.json()
    #print("Wreczamy ciasteczko mistrza dyscypliny do uzytkownikow o id:", user["name"])
    print("Wreczamy ciasteczko mistrza dyscypliny do uzytkownikow o id:", user[0]["name"])


#sposob 3
def change_list_into_conj_of_param(my_list, key="id"):
    conj_param = key + "="

    lastIteration = len(my_list)
    i = 0
    for item in my_list:
        i += 1
        if(i == lastIteration):
            conj_param += str(item)
        else:
            conj_param += str(item) + "&" + key + "="

        
    return conj_param
    
conj_param = change_list_into_conj_of_param(userIdWithMaxCompletedAmountOfTasks, "id")
#conj_param = change_list_into_conj_of_param([1,2,3])

r = requests.get("https://jsonplaceholder.typicode.com/users/", params=conj_param)
users2 = r.json()
for user in users2:
    print("Wreczamy ciasteczko mistrza dyscypliny do uzytkownikow o id:", user["name"])
"""






"""
#ĆWICZENIE - przefiltruj otwierające strony od nieotwierających się
import requests

with open ("strony.txt", "r", encoding = "UTF-8") as file:
    strona = file.readlines()
    file.close()
    
    with open ("dzialajacestrony.txt", "w", encoding = "UTF-8") as file:
        for num in range(len(strona)):
            try:
                response = requests.get(strona[num].strip())  # Remove trailing whitespace
                if response.status_code == 200:
                    print(response)
                    file.write(strona[num].strip())
                    file.write("\n")
                else:
                    print("No response")
            except:
                print("Error: ", e)
    file.close()
"""






"""
#ĆWICZENIE: Częstotliwość występowania słowa w pliku
sciezka = "tekst.txt"
slowo = "kot"

try:
    with open (sciezka, "r", encoding="UTF-8") as file:
        tekst = file.read()
        wystapienia = tekst.count(slowo)
        
    print(f"Liczba wystąpień '{slowo}' w pliku {sciezka} to {wystapienia}.")
except FileNotFoundError:
    print(f"Plik o ścieżce {sciezka} nie został znaleziony.")
except PermissionError:
    print(f"Brak uprawnień do odczytu pliku {sciezka}.")
"""






"""
#ĆWICZENIE: FileNotFoundError exception
from enum import Enum
#import os

extension = Enum("extension", ["ext1", "ext2"])

ExtensionType = {
                    extension.ext1: ".txt",
                    extension.ext2: ".csv"
                }

def read_content_of_file(path):
    try:
        with open (path, "r", encoding="UTF-8") as file:
            return file.read()
    except FileNotFoundError:
        print("Nie znaleziono pliku. ")

nameOfFile = input("Podaj nazwe pliku do otwarcia: ")

filename, file_extension = os.path.splitext(nameOfFile)
if not file_extension:
    x = None
    while x not in ["1", "2"]:
        x = input("Jaki jest to typ pliku? \n1 - .txt\n2 - .csv \n")
        if(x == "1"):
            nameOfFile += ExtensionType[extension.ext1]
        elif(x == "2"):
            nameOfFile += ExtensionType[extension.ext2]
        else:
            print("Podano niewlasciwa wartosc. ")
            continue

fileContent = read_content_of_file(nameOfFile)

print(fileContent)
"""






"""
#82. Słowo kluczowe: EXCEPT - obsługa wyjątku - ĆWICZENIE: Rozdziel imiona i nazwiska
namesandsurnames = []

with open ("imionanazwiska.txt", "r", encoding="UTF-8") as file:
    for line in file:
        namesandsurnames.append(tuple(line.replace("\n", "").split(" ")))

with open ("imiona.txt", "w", encoding="UTF-8") as file:
    for item in namesandsurnames:
        try: 
            file.write(item[0] + "\n")
        except IndexError:
            file.write("\n")

with open ("nazwiska.txt", "w", encoding="UTF-8") as file:
    for item in namesandsurnames:
        try: 
            file.write(item[1] + "\n")
        except IndexError:
            file.write("\n")
"""






"""
#73. ĆWICZENIE | GRA | Otwieranie skrzynek z losową ilością złota
#74. ĆWICZENIE | GRA | Losowanie liczb przybliżonych

import random
#from collections import Counter
from enum import Enum

def findApproximateValue(value, percentRange):
    lowestValue = value - percentRange / 100 * value
    highestValue = value + percentRange / 100 * value
    return random.randint(lowestValue, highestValue)
    
Event = Enum("Event", ["Chest", "Empty"])

eventDictionary = {
                    Event.Chest: 0.6,
                    Event.Empty: 0.4
                  }

eventList = list(eventDictionary.keys())
eventProbability = list(eventDictionary.values())

Colours = Enum("Colours", {"Green": "zielony",
                           "Orange": "pomaranczowy",
                           "Purple": "fioletowy",
                           "Gold": "zloty"
                           }
               )

chestColoursDictionary = {
                    Colours.Green: 0.75,
                    Colours.Orange: 0.2,
                    Colours.Purple: 0.04,
                    Colours.Gold: 0.01
                  }

chestColoursList = tuple(chestColoursDictionary.keys())
chestColoursProbability = tuple(chestColoursDictionary.values())

rewardsForChests = {
                    chestColoursList[reward]: (reward + 1) * (reward + 1) * 1000
                    for reward in range(len(chestColoursList))
                  }

gameLength = 5
goldAcquired = 0
percentRange = 0
print("Welcome to the game\nyou can make 5 steps. ")

while gameLength >0 :
    gameAnswer = input("Do you want to move forward? ")
    if(gameAnswer == "yes"):
        print("Great, let,s see what you got... ")
        drawnEvent = random.choices(eventList, eventProbability)[0]
        if (drawnEvent == Event.Chest):
            print("You've drawn a chest. ")
            drawnChest = random.choices(chestColoursList, chestColoursProbability)[0] 
            print("The chest you've drawn is", drawnChest.value)
            percentRange = random.randint(0, 10)
            gamerReward = findApproximateValue(rewardsForChests[drawnChest], percentRange)
            goldAcquired = goldAcquired + gamerReward
        elif(drawnEvent == Event.Empty):
            print("Unlucky no prize for you. ")
    else:
        print("You can go forward")
        continue
    
    gameLength -= 1

print("You got:", goldAcquired)
"""






"""
#72. ĆWICZENIE: Losowanie elementów bez DUPLIKATÓW - piszemy generator 6 z 49 liczb
import random
from collections import Counter
cardList = ["9", "9", "9", "9",
            "10", "10", "10", "10",
            "Jack", "Jack", "Jack", "Jack",
            "Queen", "Queen", "Queen", "Queen",
            "King", "King", "King", "King",
            "Ace", "Ace", "Ace", "Ace",
            "Joker", "Joker"]
random.shuffle(cardList)
#print(random.sample(cardList, 5))
talia = []
talia2 = []
talia3 = []
talia4 = []

def losuj_talie(talia, cardList):
    for i in range (0,5):
        karta = cardList.pop()
        talia.append(karta)
    return talia


print(losuj_talie(talia, cardList))
print(losuj_talie(talia2, cardList))
print(losuj_talie(talia3, cardList))
print(losuj_talie(talia4, cardList))
print(Counter(cardList))

#sposob 1 bez sample zadanie domowe
container = []
def choose_random_numbers(amount, total_amount):
    while len(container) < amount:
        number = random.randint(1, total_amount)
        if(number not in container):
            container.append(number)
            
    return container

random_numbers = choose_random_numbers(6, 49)
print(random_numbers)
#sposob 2
print("/////////////////////////")
def choose_random_numbers2(amount, total_amount):
    print(random.sample(range(total_amount+1 ), amount))

random_numbers = choose_random_numbers2(6, 49)

"""






"""
#ĆW: Ponumeruj zadania i pokaż je użytkownikowi | enumerate()

tasks = ["clean the kitchen", "do laundry", "pay bills"]
 
for i, tasks in enumerate(tasks, start = 1):
    print(i, tasks)
"""






"""
#ĆW: Użyj funkcji all(), aby określić, czy użytkownicy spełniają kryteria

def has_required_skills(person, skills):
    return all(skill in person["skills"] for skill in skills)

john = {
    'name': 'John Doe',
    'age': 30,
    'skills': ['Python', 'JavaScript', 'C++']
}

jane = {
    'name': 'Jane Smith',
    'age': 25,
    'skills': ['Python', 'Java']
}

required_skills = ["Python", "JavaScript"]
print(has_required_skills(john, required_skills))
print(has_required_skills(jane, required_skills))
"""






"""
#66. ĆWICZENIE: Użyj funkcji any(), aby określić, czy lista zawiera liczby parzyst
numbers1 = [1, 3, 5, 6, 7]
numbers2 = [1, 3, 5, 7, 9]
numbers3 = [2, 4, 6, 8]
def any_even(lista):
    return any([nr % 2 == 0 for nr in lista])

def all_even(lista):
    return all([nr % 2 == 0 for nr in lista])

print(any_even(numbers1))
print(any_even(numbers2))

if(any_even(numbers1)):
    print("Tak")
else:
    print("nie")
print("/////////////////////////////////////")
#Cwiczenie

def knows_language(person):
    return (
        any([p == 'Python' for p in person['skills']]) and
        any([p == 'JavaScript' for p in person['skills']])
    )

john = {
    'name': 'John Doe',
    'age': 30,
    'skills': ['Python', 'JavaScript', 'C++']
}

jane = {
    'name': 'Jane Smith',
    'age': 25,
    'skills': ['Python', 'Java']
}
print(knows_language(john))
print(knows_language(jane))
"""






"""
#Do zorzumienia o przypisaniach
a = [7,8,9]
def append_element(a, e):
    print(id(a))
    a.append(e)
    print(id(a))
    print(a)

append_element(a,4)
"""






"""
#ĆWICZENIE: Liczenie sumy wszystkich argumentów
def count(*args):
    suma = 0
    for arg in args:
        suma += arg
    return suma

print(count(2, 4, 1, 2, 4, 5, 10))
"""






"""
#59. ĆWICZENIE: Sprawdzenie, czy istnieje element wewnątrz zbioru vs listy
import time
 
def function_performance(func, how_many_times = 1, **arg):
    sum = 0
    print(arg.get("what_value"))
    print(arg.get("container"))
    for i in range(0, how_many_times):
        start = time.perf_counter()
        func(**arg)
        end = time.perf_counter()
        sum = sum + (end - start)
    return sum

setContainer = {i for i in range(1000)}
listContainer = [i for i in range(1000)]

def is_element_in(what_value, container):
    if what_value in container:
        return True
    else:
        return False

print(function_performance(is_element_in, 300, what_value = 500, container = setContainer))
print(function_performance(is_element_in, how_many_times = 300, what_value = 500, container = listContainer))
#Nienazwane * nazwane **
"""






"""
#54.55.56.57. ĆWICZENIE: Suma wszystkich liczb do podanej wartości | UWAGA bazujemy na tym ćw.
import time
def sumuj_do(liczba):
    suma = 0

    for liczba in range(1, liczba+1):
        suma = suma + liczba

    return suma

def sumuj_do2(liczba):
    return sum([liczba for liczba in range(1, liczba+1)])

def sumuj_do3(liczba):
    return (1 + liczba)/ 2 * liczba

def finish_timer(start):
    end = time.perf_counter()
    return end - start

def function_performance(func, arg, how_many_times = 1):
    sum = 0
    
    for i in range(0, how_many_times):
        start = time.perf_counter()
        func(arg)
        end = time.perf_counter()
        sum = sum + (end - start)
        
    return sum

def show_message(message):
    print(message)

print(function_performance(sumuj_do, 5000000, 4))
print(function_performance(sumuj_do2, 5000000))
print(function_performance(sumuj_do3, 5000000))

#inny sposob
def finish_timer(start):
    end = time.perf_counter()
    return end - start

print("\nMetoda pierwsza")
start = time.perf_counter()
print("Wynik: ", sumuj_do(50000000))
print("Czas: ", finish_timer(start))

print("\nMetoda druga")
start = time.perf_counter()
print("Wynik: ", sumuj_do2(50000000))
print("Czas: ", finish_timer(start))

print("\nMetoda trzecia")
start = time.perf_counter()
print("Wynik: ", sumuj_do3(50000000))
print("Czas: ", finish_timer(start))
"""







"""
#ĆWICZENIE: Dynamiczny Słownik z Definicjami + funkcje

def dodajDefinicje(klucz, definicja, definicje):
    definicje[klucz] = definicja
    print("Dodano definicje")

def znajdzDefinicje(klucz, definicje):
    if klucz in definicje:
        print(definicje[klucz])
    else:
        print("Nie znaleziono definicji o nazwie", klucz)
        
def usunDefinicje(klucz, definicje):
    if klucz in definicje:
        del(definicje[klucz])
        print("Usunieto definicje o nazwie: ", klucz)
    else:
        print("Nie znaleziono definicji o nazwie", klucz)
    
def zakoncz():
    print("Do widzenia! ")

definicje = {}

while(True):
    print("\n1: Dodaj definicje")
    print("2: Znajdz definicje")
    print("3: Usun definicje")
    print("4: Zakoncz")

    menu = input("\nCo chcesz zrobic? ")

    if(menu == '1'):
        klucz = input("Podaj klucz (slowo) do zdefiniowania: ")
        definicja = input("Podaj definicje: ")
        dodajDefinicje(klucz, definicja, definicje)
    elif(menu == '2'):
        klucz = input("Czego szukasz? ")
        znajdzDefinicje(klucz, definicje)
    elif(menu == '3'):
        klucz = input("Ktory klucz chcesz usunac ")
        usunDefinicje(klucz, definicje)
    elif(menu == '4'):
        zakoncz()
        break
    else:
        print("\nCzytac nie umiesz? Od 1 do 4")
"""






"""
#ĆWICZENIE: Sumowanie liczb dodatnich w liście
def lista_liczb_suma(numbers):
    sum = 0
    for number in numbers:
        if(number > 0):
            sum += number
    return sum

numbers = []

i = int(input("Podaj ile liczb w liscie chcesz? "))
for x in range(i):
    liczba = int(input("Podaj liczbę: "))
    numbers.append(liczba)
    
print("Suma dodatnich niezerowych liczb = ", lista_liczb_suma(numbers)
#Wersja c++-sowa (rozwiazanie python ale tak jak w c++ bym robil) zakomentuj 1
#rozni sie tym od kodu wyzej z po podaniu i dynamicznie alokujemy liste numbers
#o i-miejsach wypelnionych 0 czyli dla i = 3 [0,0,0] a potem w petli przypisujemy
#np numbers[0]=1 numbers[1]=2 numbers[2]=-3 
def lista_liczb_suma(numbers):
    sum = 0
    for number in numbers:
        if(number > 0):
            sum += number
    return sum

i = int(input("Podaj ile liczb w liscie chcesz? "))
numbers = [0]*i #dynamiczna alokacja

for x in range(i):
    numbers[x] = int(input("Podaj liczbę: ")) #tu nie liczba i potem numbers.append
    
print("Suma dodatnich niezerowych liczb = ", lista_liczb_suma(numbers))
#"""






"""
#ĆWICZENIE: Program liczący powierzchnie figur
import math

def pole_kwadratu(a):
    return a * a

def pole_prostokata(a, b):
    return a * b

def pole_kola(r):
    return math.pi * r ** 2

def pole_trojkata(a, h):
    return 0.5 * a * h

def pole_trapezu(a, b, h):
    return (a + b) / 2 * h

menu = 0

while menu != 6:
    print("\n1 - Pole kwadratu. ")
    print("2 - Pole prostokata. ")
    print("3 - Pole kola. ")
    print("4 - Pole trojkata. ")
    print("5 - Pole trapezu. ")
    print("6 - Zakoncz program. ")
    menu = input("Podaj jakie pole chcesz policzyć? ")

    if(menu == '1'):
        a = float(input("Podaj dlugosc "))
        print("\nPole kwadratu wynosi ", pole_kwadratu(a))
    elif(menu == '2'):
        a = float(input("Podaj dlugosc "))
        b = float(input("Podaj szerokosc "))
        print("\nPole prostokata wynosi ", pole_prostokata(a, b))
    elif(menu == '3'):
        r = float(input("Podaj promien "))
        print("\nPole kola wynosi ", pole_kola(r))
    elif(menu == '4'):
        a = float(input("Podaj dlugosc podstawy "))
        h = float(input("Podaj wysokosc "))
        print("\nPole trojkata wynosi ", pole_trojkata(a, h))
    elif(menu == '5'):
        a = float(input("Podaj rozmiar pierwszej podstawy "))
        b = float(input("Podaj rozmiar drugiej podstawy "))
        h = float(input("Podaj wysokosc "))
        print("\nPole trapezu wynosi ", pole_trapezu(a, b, h))
    elif(menu == '6'):
        print("\nNarazie! ")
        break
    else:
        print("\nPodano niewlasciwa liczbe")
        continue
"""






"""
#44. ĆWICZENIE: Wyszukanie liczb podzielnych przez 7, ale niepodzielnych przez 5
numbers = [
    number
    for number in range(2, 471)
    if(number % 7 == 0 and number % 5 != 0 ) # mozna zapisac tez jeden if pod drugim bez "and"
    ]
print(numbers)
"""






"""
#43. Wyrażenie zbioru
names = {"Kuba", "bartek", "Jula", "szymon", "Bronislaw", "ania"}

names = {
    name.capitalize()
    for name in names
    if(name.capitalize().startswith("B") == False)
    }

print(names)
"""






"""
#42. Wyrażenia (formuły) słownikowe | Transformacja Słownika

names = {"Kuba", "Julia", "Karolina", "Maciek"}

newNames = {
    len(name) : name
    for name in names
    if name.startswith("K")
    }

print(names)
print(newNames)

numbers = [1, 2, 3, 4, 5, 6, 7]

newNumbers = {
    number : number**3
    for number in numbers
    if((number+numbers[6]) >= 10)
    }

print(numbers)
print(newNumbers)

temp = {"t1": -20, "t2": -15, "t3": 0, "t4": 12, "t5": 24}
newTemp = {
    key : (t*1.8)+32
    for key, t in temp.items()
    if t > -5
    if t < 20
    }

print(temp)
print (newTemp)
"""






"""
#41. Wyrażenia generujące (generator jest () a lista [] nazwa wywalone)
import sys

squaredGenerator = [element**2
                   for element in range(101)
                   ]

i = 0
for item in squaredGenerator:
    print(i, "         ", item)
    i += 1

suma = sum(squaredGenerator)
print(suma)
"""






"""
#39. ĆWICZENIE: Dynamiczny Słownik z Definicjami

definicje = {}

while(True):
    print("\n1: Dodaj definicje")
    print("2: Znajdz definicje")
    print("3: Usun definicje")
    print("4: Zakoncz")

    menu = input("\nCo chcesz zrobic? ")

    if(menu == '1'):
        klucz = input("Podaj klucz (slowo) do zdefiniowania: ")
        definicja = input("Podaj definicje: ")
        definicje[klucz] = definicja
        print("Dodano definicje")
    elif(menu == '2'):
        klucz = input("Czego szukasz? ")
        if klucz in definicje:
            print(definicje[klucz])
        else:
            print("Nie znaleziono definicji o nazwie", klucz)
    elif(menu == '3'):
        klucz = input("Ktory klucz chcesz usunac ")
        if klucz in definicje:
            del(definicje[klucz])
            print("Usunieto definicje o nazwie: ", klucz)
        else:
            print("Nie znaleziono definicji o nazwie", klucz)
    elif(menu == '4'):
        print("Do widzenia! ")
        break
    else:
        print("\nCzytac nie umiesz? Od 1 do 4")
"""





        
"""
#30. ĆWICZENIE - Przekaż dostęp do sekcji kodu tylko uprawnionym osobom
imiona = ["Arkadiusz", "Wiola", "Antek"]
imie = input("Podaj imie: ")
imie = imie.capitalize()

if(imie in imiona):
    print("Masz dostep")
else:
    print("Brak dostepu")
#KONIEC
    
#Sposob II
if(imie == "Arkadiusz" or imie == "Wiola" or imie == "Antek"):
    print("Masz dostep")
else:
    print("Brak dostepu")
"""






"""
#27 Zgadnij Liczbe
szukana = 40
zgadywana = 0

while zgadywana != szukana :
    zgadywana = int(input("Podaj liczbe: "))
    if(zgadywana > szukana):
        print("za duzo")
        continue
    elif(zgadywana < szukana):
        print("za malo")
        continue
print("Gratulacje")
"""






"""
#Zadanie 1: Dodawanie liczb parzystych podanych przez użytkownika
i = 0

wynik = 0 

while i < 3:
    x = int(input("Podaj liczbe dodatnia parzysta: "))
    if(x % 2 == 0 and x > 0):
        wynik += x
    else:
        print("Jeden z warunkow nie jest spelniony")
        continue
    print("Twoj wynik to", wynik)
    i += 1
"""






"""
#25. Pętla for
#Wydrukuj liczby 0-200 podzielne przez 5 a nie przez 7 
for i in range(205):
    if (i%5 == 0 and i%7 != 0):
        print(i)
"""






"""
25. Pętla for

liczba = 100
while liczba >= 0:
    print(liczba)
    liczba -= 1
"""






"""
#19. ĆWICZENIE: Wartość bezwzględna z liczby
a = int(input("Podaj liczbe"))

if(a < 0): #Sposob 1 
    a = -a

print(a)

#KONIEC I SPOSOB 2 TO
a = int(input("Podaj liczbe"))
print(abs(a))
"""






"""
18. Instrukcja warunkowa If oraz WCIĘCIA | UWAGA! | Python inaczej odbiera wcięcia
wybor = input("* - Mnozenie, / - Dzielenie, + - Dodawanie, - - Odejmowanie, ^ - Potega, % - Modulo: \n")

if(wybor == '*' or wybor == '/' or wybor == '+' or wybor == '-' or wybor == '^' or wybor == '%'):
    a = int(input("Pierwsza liczba: "))
    b = int(input("Druga liczba: "))

    if(wybor == '*'):
        print(a * b)
    elif(wybor == '/'):
        if(b == 0):
            print("Nie da sie")
        else:
            print(a / b)
    elif(wybor == '+'):
        print(a + b)
    elif(wybor == '-'):
        print(a - b)
    elif(wybor == '^'):
        print(a ** b)
    elif(wybor == '%'):
        f(b == 0):
            print("Nie da sie")
        else:
            print(a % b)
else:
    print("Podano niewlasciwy znak")
"""






"""
16. ĆWICZENIE: Pobieranie i formatowanie danych wprowadzonych przez użytkownika
imie = input("Podaj Imie")
wiek =  int(input("Podaj wiek"))
kolor =  input("Podaj kolor")

print("Hej " + imie + ", Twój ulubiony kolor to " + kolor + ".\nMasz " , wiek , " lat.\nZa rok bedziesz mial" , wiek + 1 , " lat.")
"""
