import pickle

label = ['exciting', 'stomachache', 'butt_scooting', 'stressed', 'very_aggressive']
file = open('./data/label.txt', 'wb')
pickle.dump(label, file)
file.close()