import pandas as pd
from keras.models import Sequential as se
from keras.layers import Dense
from  keras.optimizers import RMSprop

df = pd.read_csv('wine.csv')
y= df['Class']
y_category = pd.get_dummies(y)
x = df.drop('Class', axis=1)


model = se()
model.add(Dense(units=14, input_shape=(13,), activation='relu', kernel_initializer='he_normal' 
model.add(Dense(units=10, activation='relu', kernel_initializer='he_normal' )) 
model.add(Dense(units=3, activation='softmax' )) 
model.compile(optimizer=RMSprop(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x, y_category, epochs=50)
result = history.history['accuracy'][-1]*100
print("Accuracy of the model is : ", result)