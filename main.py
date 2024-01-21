#!/usr/bin/env python3
import enum
import typing
import random

class EnumValue:
  def __init__(self, option_name: str):
    self.option_name = option_name
  def __str__(self):
    return self.option_name
  def __repr__(self):
    return self.__str__()
class GenderValue(EnumValue):
  def __init__(self, option_name: str, he: str, his: str, him: str):
    super().__init__(option_name)
    self.he = he
    self.his = his
    self.him = him

class Gender(enum.Enum):
  MALE=GenderValue("male", "he", "his", "him")
  '''
  Examples:
  ```
  option = Gender.MALE
  option.name # "MALE"
  option.value # GenderValue(...)
  option.value.option_name # "male"
  ```
  '''
  FEMALE=GenderValue("female", "she", "her", "her")

class SexualityValue(EnumValue):
  def __init__(self, option_name: str):
    super().__init__(option_name)

class Sexuality(enum.Enum):
  HETEROSEXUAL=SexualityValue("heterosexual")
  HOMOSEXUAL=SexualityValue("homosexual")
class SmokerValue(EnumValue):
  def __init__(self, option_name: str):
    super().__init__(option_name)
class Smoker(enum.Enum):
  NON_SMOKER=SmokerValue("non-smoker")
  SMOKER=SmokerValue("smoker")

def get_enum_value(enum_class: type[EnumValue], name: str) -> EnumValue | None:
  '''
  Example usage: `get_enum_value(Gender, "female")` will return `Gender.FEMALE`
  '''
  for option in enum_class:
    if name == str(option.value):
      return option
  return None

# choices(population, weights=None, *, cum_weights=None, k=1)
def rand_enum(enum_class: typing.Type[enum.Enum], num_elements):
  return random.choices(list(enum_class), weights=None, k=num_elements)

class User:
  name_pool = ('John','Andy','Joe')
  def __init__(self, is_bot: bool, name: str, gender: Gender, sexuality: Sexuality, age: int, hobby: str, username: str, user_id: int, chat_id: int,
               smoking: Smoker):
    self.is_bot = is_bot
    self.name = name
    self.gender = gender
    self.sexuality = sexuality
    self.age = age
    self.hobby = hobby
    self.last_matched_with: User | None = None
    self.current_matched_with: User | None = None
    self.username = username
    self.user_id = user_id
    self.chat_id = chat_id
    # other factors:
    self.smoking = smoking

  @classmethod
  def exists(cls, user_id: int) -> bool:
    return user_id in all_users

  @classmethod
  def generate_bot(cls, num_elements) -> list["User"]:
    names = random.choices(User.name_pool, weights = None, k = num_elements)
    genders = rand_enum(Gender, num_elements)
    sexualities = rand_enum(Sexuality, num_elements)
    ages = random.choices(range(18, 61), weights = None, k = num_elements)
    users = []
    for name, gender, sexuality, age in zip(names, genders, sexualities, ages):
      users.append(User(True, name, gender, sexuality, age, "hobby", "bot", -1, -1, False))
    return users

  def __str__(self):
    return f"({self.name}, {self.age}, {self.hobby})"

  def description_for_llm(self):
    return f"""
    {self.name} is {self.gender.value}, {self.age} years of age.
    {self.gender.value.he} is {self.sexuality.value} and {self.gender.value.his} favorite thing to do is {self.hobby}.
    He/She prefers to {'not' if self.smoking else ''} smoke.
    """



class Factor:
  def compute_partial_compatibility_score(self, u1: User, u2: User) -> tuple[float, float]:
    '''
    Returns a tuple `[u1_likes_u2, u2_likes_u1]`

    `u1_likes_u2` represents how much `u1` would like `u2`

    `u2_likes_u1` represents how much `u2` would like `u1`
    '''
    raise Exception("abstract method")

class AgeFactor(Factor):
  def compute_age_score(self, u1: User, u2: User) -> float :
    return 1 / (abs(u1.age - u2.age) + 1)
  def compute_partial_compatibility_score(self, u1: User, u2: User) -> tuple[float, float]:
    res = self.compute_age_score(u1, u2);
    return [res, res]

class HobbyFactor(Factor):
  def compute_hobby(self, u1: User, u2: User):
    return 1 if u1.hobby == u2.hobby else 0

  def compute_partial_compatibility_score(self, u1: User, u2: User) -> tuple[float, float]:
    res = self.compute_hobby(u1, u2);
    return [res, res]

class SmokingFactor(Factor):
  def compute_smoking(self, u1: User, u2: User):
    return 1 if u1.smoking == u2.smoking else 0

  def compute_partial_compatibility_score(self, u1: User, u2: User) -> tuple[float, float]:
    res = self.compute_smoking(u1, u2);
    return [res, res]

def compute_pairwise_compatibility_score(u1: User, u2: User, factors: list[Factor]):
  score1, score2 = 0, 0
  for factor in factors:
    s1, s2 = factor.compute_partial_compatibility_score(u1, u2)
    score1 += s1
    score2 += s2
  return score1, score2

male_hetero_users = [
  #  is_bot: bool, name: str, gender: Gender, sexuality: Sexuality, age: int, hobby: str, username: str, user_id: int, chat_id: int):
  User(False, "Merlin", Gender.MALE, Sexuality.HETEROSEXUAL, 91, 'complaining', "merlin", 1, -1, Smoker.SMOKER),
  User(False, "Stark", Gender.MALE, Sexuality.HETEROSEXUAL, 34, 'french', "stark", 2, -1, Smoker.SMOKER),
  User(False, "Dick", Gender.MALE, Sexuality.HETEROSEXUAL, 18, 'clubbing', "dick", 3, -1, Smoker.NON_SMOKER),
  User(False, "Charlie", Gender.MALE, Sexuality.HETEROSEXUAL, 23, 'tennis', "charlie", 4, -1, Smoker.NON_SMOKER),
  User(False, "Authur", Gender.MALE, Sexuality.HETEROSEXUAL, 45, 'japanese', "authur", 5, -1, Smoker.NON_SMOKER)
]

female_hetero_users = [
  User(False, "Nancy", Gender.FEMALE, Sexuality.HETEROSEXUAL, 97, 'cats', "nancy", 6, -1, Smoker.NON_SMOKER),
  User(False, "Winnie", Gender.FEMALE, Sexuality.HETEROSEXUAL, 33, 'badminton', "winnie", 7, -1, Smoker.NON_SMOKER),
  User(False, "Stefanie", Gender.FEMALE, Sexuality.HETEROSEXUAL, 19, 'tennis', "stefanie", 8, -1, Smoker.SMOKER),
  User(False, "Charmaine", Gender.FEMALE, Sexuality.HETEROSEXUAL, 27, 'tennis', "charmaine", 9, -1, Smoker.SMOKER),
  User(False, "Iko", Gender.FEMALE, Sexuality.HETEROSEXUAL, 44, 'japanese', "iko", 10, -1, Smoker.SMOKER)
]

male_homo_users = [
  User(False, "Moby", Gender.MALE, Sexuality.HOMOSEXUAL, 93, 'badminton', "moby", 11, -1, Smoker.SMOKER),
  User(False, "Enola", Gender.MALE, Sexuality.HOMOSEXUAL, 37, 'french', "enola", 12, -1, Smoker.SMOKER),
  User(False, "Rod", Gender.MALE, Sexuality.HOMOSEXUAL, 18, 'tennis', "rod", 13, -1, Smoker.NON_SMOKER),
  User(False, "Prince", Gender.MALE, Sexuality.HOMOSEXUAL, 28, 'tennis', "prince", 14, -1, Smoker.NON_SMOKER),
  User(False, "Bishop", Gender.MALE, Sexuality.HOMOSEXUAL, 47, 'japanese', "bishop", 15, -1, Smoker.NON_SMOKER)
]

female_homo_users = [
  User(False, "Enolass", Gender.FEMALE, Sexuality.HOMOSEXUAL, 99, 'badminton', "enolass", 16, -1, Smoker.NON_SMOKER),
  User(False, "Princess", Gender.FEMALE, Sexuality.HOMOSEXUAL, 31, 'badminton', "princess", 17, -1, Smoker.NON_SMOKER),
  User(False, "Kat", Gender.FEMALE, Sexuality.HOMOSEXUAL, 19, 'tennis', "kat", 18, -1, Smoker.SMOKER),
  User(False, "Joey", Gender.FEMALE, Sexuality.HOMOSEXUAL, 21, 'tennis', "joey", 19, -1, Smoker.SMOKER),
  User(False, "Sam", Gender.FEMALE, Sexuality.HOMOSEXUAL, 42, 'japanese', "sam", 20, -1, Smoker.SMOKER)
]

def split_into_groups(group: list[User]):
  random.shuffle(group)
  group1 = group[:len(group) // 2]
  group2 = group[len(group) // 2:]

  return (group1, group2)

def make_two_groups_equally_sized(group1: list[User], group2: list[User]):
  while len(group1) < len(group2):
    users = User.generate_bot(1)
    group1.append(users[0])
  while len(group2) < len(group1):
    users = User.generate_bot(1)
    group2.append(users[0])

def generate_match_matrices(group1: list[User], group2: list[User]):
  make_two_groups_equally_sized(group1, group2)
  n = len(group1)
  mat1 = [[None] * n for _ in range(n)]
  mat2 = [[None] * n for _ in range(n)]
  for i1, u1 in enumerate(group1):
    for i2, u2 in enumerate(group2):
      u1_to_u2, u2_to_u1 = compute_pairwise_compatibility_score(u1, u2, [AgeFactor(), HobbyFactor()])
      mat1[i1][i2] = (i2 + n, u1_to_u2)
      mat2[i2][i1] = (i1, u2_to_u1)
  for row in mat1:
    row.sort(key = lambda tup: tup[1])
  for row in mat2:
    row.sort(key = lambda tup: tup[1])
  mat1_processed = list(map(lambda row: list(map(lambda tup: tup[0], row)), mat1))
  mat2_processed = list(map(lambda row: list(map(lambda tup: tup[0], row)), mat2))
  mat_processed = mat1_processed + mat2_processed
  return mat_processed

# Python3 program for stable marriage problem

# Number of Men or Women

def wPrefersM1OverM(prefer, w, m, m1, N):
	for i in range(N):
		if (prefer[w][i] == m1):
			return True
		if (prefer[w][i] == m):
			return False

def stableMarriage(prefer):
	N = len(prefer) // 2
	wPartner = [-1 for i in range(N)]
	mFree = [False for i in range(N)]

	freeCount = N

	# While there are free men
	while (freeCount > 0):
		m = 0
		while (m < N):
			if (mFree[m] == False):
				break
			m += 1
		i = 0
		while i < N and mFree[m] == False:
			w = prefer[m][i]
			if (wPartner[w - N] == -1):
				wPartner[w - N] = m
				mFree[m] = True
				freeCount -= 1

			else:
				m1 = wPartner[w - N]
				if (wPrefersM1OverM(prefer, w, m, m1, N) == False):
					wPartner[w - N] = m
					mFree[m] = True
					mFree[m1] = False
			i += 1

	# Print solution
	# print("Woman ", " Man")
	# for i in range(N):
	# 	print(i + N, "\t", wPartner[i])
	return wPartner

# # Driver Code
# prefer = [
# 		 # Men to woman
# 		[7, 5, 6, 4],
# 		[5, 4, 6, 7],
# 		[4, 5, 6, 7],
# 		[4, 5, 6, 7],

# 		# Woman to men
# 		[0, 1, 2, 3], [0, 1, 2, 3],
# 		[0, 1, 2, 3], [0, 1, 2, 3]
# ]

# def reversePerferences(prefer):
#   reverse = [prefer[i][::-1] for i in range(len(prefer))]
#   return reverse

def do_matches(group1: list[User], group2: list[User]):
  mat = generate_match_matrices(group1, group2)

  matching = stableMarriage(mat)
  '''matching is a mapping from group2 to group1'''
  for group2_idx, group1_idx in enumerate(matching):
    group1[group1_idx].last_matched_with = group1[group1_idx].current_matched_with
    group2[group2_idx].last_matched_with = group2[group2_idx].current_matched_with
    group1[group1_idx].current_matched_with = group2[group2_idx]
    group2[group2_idx].current_matched_with = group1[group1_idx]
def just_do_it():
  do_matches(male_hetero_users, female_hetero_users)
  do_matches(*split_into_groups(female_homo_users))
  do_matches(*split_into_groups(male_homo_users))

def print_matchings(users: list[User]):
  for user in users:
    print(f"{user} - {user.current_matched_with}")

all_users = {}
import os
from langchain_community.llms import HuggingFaceHub


class LLM:
  def __init__(self):
    model_string = "mistralai/Mistral-7B-Instruct-v0.2"
    self.chat = []
    self.llm = HuggingFaceHub(repo_id=model_string, model_kwargs={"temperature": 0.5, "max_length":64,"max_new_tokens":512})

  def get_date(self, user_one: User, user_two: User):
    instruction = f"""
    Hello, I am going to tell you about two people I want to connect together.
    {user_one.description_for_llm()}
    {user_two.description_for_llm()}

    Introduce {user_one.name} to {user_two.name} in a sacarstic tone.
    Highlight how incompatible {user_two.name} is to {user_one.name} in the process.

    Then, suggest the worst date activity (suggest only one) for these two people in a sarcastic tone.

    Please phrase this message such that it can be sent directly to the person who is being matchmade.

    There is an example below

    Dear (insert user name here),

    (insert user name here), meet your perfect match, (insert name of match here). She's only 1 year older than your granddaughter, but don't let that deter you! She's got a thing for all things Japanese, while you're out there hooping. But hey, opposites attract, right?

    I'm sure she'll absolutely love hooping with you at a cigarette factory. After all, what better way to spend a romantic evening than inhaling secondhand smoke while twirling around in circles? I'm sure you'll both have a blast!

    Best,
    Your Matchmaker.
    """
    return self.llm.invoke(instruction)


  def get_activity(self, user_one: User, user_two: User):
    instruction = f"""<INSTRUCT>
    Hello, I am going to tell you about two people I want to matchmake together.
    {user_one.description_for_llm()}
    {user_two.description_for_llm()}

    Suggest the worst date activity (suggest only one) for these two people in a sarcastic tone. Please phrase this message such that it can be sent directly to the person who is being matchmade.
    <INSTRUCT>"""
    return self.llm.invoke(instruction)

  def get_sarcastic_response(self, activity_prompt):
     instruction = f"""Based on this activity prompt '{activity_prompt}, make the worst date activity proposal sarcastic and doomish sounding to users in this format: You are challenged to go on the date below: (insert the activity proposal here)"""
     return self.llm.invoke(instruction)

import telebot

TELEGRAM_API_KEY=os.environ['UNHINGED_TOKEN']
bot = telebot.TeleBot(TELEGRAM_API_KEY)
bot.get_me()
import telebot
from telebot import types
factors = [(Gender, "enum"), (Sexuality, "enum"), ("Age", "int"), ("Favourite Hobby", "str"), (Smoker, "enum")]
## Function to create keyboard
def get_keyboard(options):
    keyboard = types.ReplyKeyboardMarkup()
    for option in options:
      keyboard.add(types.KeyboardButton(str(option.value)))
    return keyboard

def yes_no_keyboard():
  keyboard = types.ReplyKeyboardMarkup()
  keyboard.add(types.KeyboardButton("YES"))
  keyboard.add(types.KeyboardButton("NO"))

  return keyboard

# Function to handle button clicks
@bot.callback_query_handler(func=lambda call: call.data is not None)
def button_click(call):
    button_clicked = call.data
    bot.answer_callback_query(call.id)
    bot.send_message(call.message.chat.id, text=f"Button {button_clicked} clicked!")

def init_bot():
  BOT_TOKEN = os.environ['UNHINGED_TOKEN']
  bot = telebot.TeleBot(BOT_TOKEN)

  @bot.message_handler(commands=['start'])
  def on_start(message):
    if User.exists(message.from_user.id):
      bot.reply_to(message, "Want to find another match? Send /find_match to find a match!")
      return
    bot.reply_to(message, "Ready for the worst date ever? Send '/preferences' if you are")

  @bot.message_handler(commands=['preferences'])
  def build_profile(message):
    if User.exists(message.from_user.id):
      bot.reply_to(message, "You've already indicated your preferences")
      return
    start_question = "Awesome, are you ready to start answering questions on your preferences?\nChoose an option using the reply keyboard provided"
    reply = bot.send_message(message.chat.id, start_question, reply_markup=yes_no_keyboard())
    bot.register_next_step_handler(reply, check_ready)

  def check_ready(message):
    if message.text == "YES":
        reply = bot.send_message(
        message.chat.id, "Great! First question, what is your name?", reply_markup=types.ReplyKeyboardRemove())
        bot.register_next_step_handler(
        reply, get_preferences, 0, factors, [])
    else:
      bot.reply_to(message, "Awwww, maybe next time!")

  def save_preferences(message, preferences):
      user = User(False, message.from_user.first_name, preferences[0], preferences[1], preferences[2], preferences[3], message.from_user.username, message.from_user.id, message.chat.id, preferences[4])
      print(preferences[0], preferences[0] == Gender.MALE, preferences[1], preferences[1] == Sexuality.HETEROSEXUAL)
      if preferences[0] == Gender.MALE and preferences[1] == Sexuality.HETEROSEXUAL:
        male_hetero_users.append(user)
      elif preferences[0] == Gender.MALE and preferences[1] == Sexuality.HOMOSEXUAL:
        male_homo_users.append(user)
      elif preferences[0] == Gender.FEMALE and preferences[1] == Sexuality.HETEROSEXUAL:
        female_hetero_users.append(user)
      else:
        female_homo_users.append(user)
      all_users[message.chat.id] = user
      bot.send_message(message.chat.id, "Your preferences have been saved. Send /find_match if you are ready for your next dating adventure")

  def get_preferences(message, i, factors, answers):
    if i == 0:
      bot.send_message(
          message.chat.id, f"Awesome! Hi {message.text}", parse_mode="Markdown")
    if i > 0 and i <= len(factors):
      prev_factor, prev_factor_type = factors[i-1]
      if prev_factor_type == "enum":
        enum_value = get_enum_value(prev_factor, message.text)
        if not enum_value:
          bot.register_next_step_handler(message, get_preferences, i, factors, answers)
          return
        answers.append(enum_value)
      elif prev_factor_type == "bool":
        answers.append(bool(message.text))
      elif prev_factor_type == "int":
        answers.append(int(message.text))
      else:
        answers.append(message.text)
    if i <= len(factors) - 1:
      factor, factor_type = factors[i]

      if factor_type == "enum":
        factor_text = str(factor.__name__)
        if factor_text == "Smoker":
          text = f"Please indicate your Smoking Habits\nChoose an option using the reply keyboard provided:"
        else:
          text = f"Please indicate your {factor_text}\nChoose an option using the reply keyboard provided:"
        reply = bot.send_message(
          message.chat.id, text, reply_markup=get_keyboard(factor))
      else:
        text = f"Please indicate your {factor}\n"
        reply = bot.send_message(
          message.chat.id, text, reply_markup=types.ReplyKeyboardRemove())
      bot.register_next_step_handler(reply, get_preferences, i + 1, factors, answers)
    else:
      reply = bot.send_message(
          message.chat.id, f"Good job! You're done. Would you like us to save your preferences?\nChoose an option using the reply keyboard provided:", reply_markup=yes_no_keyboard())
      bot.register_next_step_handler(reply, save_preferences, answers)
  def up_for_date(message):
    user1_id = message.from_user.id
    user1 = all_users[user1_id]

    user2: User = user1.current_matched_with
    if message.text == "NO":
      bot.send_message(message.chat.id, "Aww... What a missed opportunity!!!!")

      if not user2:
        rejected_chat_id = user2.chat_id
        bot.send_message(rejected_chat_id)
      return
    if not user2:
      return
    news_text = f"Breaking news -- {user1.name} is going out on a date with {user2.name}!"
    for _, v in all_users.items():
      user_chat_id = v.chat_id
      if user_chat_id != - 1:
        bot.send_message(user_chat_id, news_text)

  @bot.message_handler(commands=['find_match'])
  def match(message):
    if message.from_user.id not in all_users:
      bot.send_message(message.chat.id, "You have not registered. Please use `/preferences` to begin.")
      return
    just_do_it()
    for _, user1 in all_users.items():
      user2 = user1.current_matched_with
      reply = bot.send_message(user1.chat_id,
      f"""
      Woohoo! You were matched with @{user2.username}!
      {llm.get_date(user1, user2)}


      Are you up for the date?
      """, reply_markup=yes_no_keyboard())
      bot.register_next_step_handler(reply, up_for_date)

  return bot
bot = init_bot()
bot.infinity_polling()
