import streamlit as st
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, placeholder, start_time):
        super().__init__()
        self.placeholder = placeholder
        self.start_time = start_time

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        self.placeholder.text(f"Epoch {epoch+1}/500 - Waktu berlalu: {elapsed_time:.2f} detik")

def generate_data(a, b, n=1000):
    X = np.random.randint(1, 11, size=n)
    y = a * X + b
    return X, y

def main():
    st.set_page_config("Linear Function Prediction")
    
    # Menu sidebar
    menu = st.sidebar.selectbox("Pilih Menu", ["Home", "About"])
    
    if menu == "Home":
        st.markdown("<h1>Prediksi Fungsi Linear Menggunakan Neural Network Sederhana</h1>", unsafe_allow_html=True)
        st.markdown("<p>Selamat datang di aplikasi prediksi fungsi linear menggunakan neural network sederhana. Aplikasi ini memungkinkan Anda untuk menentukan parameter dari fungsi linear dan melihat bagaimana model memprediksi nilai berdasarkan parameter tersebut.</p>", unsafe_allow_html=True)
        st.markdown("<p>Note : Jika hasil output dari neural network lebih 1-2 angka atau kurang harap dimaklumi.</p>", unsafe_allow_html=True)
        
        a = st.sidebar.number_input("Masukan nilai a (1 - 10)", 1, 10, value=1)
        b = st.sidebar.number_input("Masukan nilai b (1 - 10)", 1, 10, value=1)
        
        X_train, y_train = generate_data(a, b)
        
        input_tebakan = st.number_input(f"Masukan angka untuk di prediksi dengan fungsi ({a}x + {b})", 1)

        if st.button("Predict"):
            with st.spinner("Mohon sebentar, model sedang di latih"):
                progress_placeholder = st.empty()
                metrics_placeholder = st.empty()

                start_time = time.time()
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(1, input_shape=[1])
                ])
                model.compile(optimizer='sgd', loss="mean_squared_error", metrics=["mean_absolute_error"])

                time_history = TimeHistory(progress_placeholder, start_time)

                history = model.fit(X_train, y_train, epochs=500, verbose=1, callbacks=[time_history])

                hasil = np.ceil(model.predict(np.array([input_tebakan])))[0][0]
                total_time = time.time() - start_time

                loss = history.history['loss'][-1]
                mae = history.history['mean_absolute_error'][-1]

                st.write(f"Prediksi: {hasil}")
                st.write(f"Waktu pelatihan total: {total_time:.2f} detik")
                st.write(f"Loss terakhir: {loss:.4f}")
                st.write(f"Mean Absolute Error (MAE) terakhir: {mae:.4f}")

                st.subheader("Grafik Metrik Pelatihan")

                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Loss')
                ax.plot(history.history['mean_absolute_error'], label='Mean Absolute Error (MAE)')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.set_title('Loss dan MAE selama Pelatihan')
                ax.legend()
                st.pyplot(fig)

    elif menu == "About":
        st.markdown("<h1>About</h1>", unsafe_allow_html=True)
        st.markdown("<p>**Linear Function Detector** adalah aplikasi web yang dibuat untuk mendemonstrasikan penggunaan neural network sederhana dalam memprediksi hasil dari fungsi linear.</p>", unsafe_allow_html=True)
        st.markdown("<p>**Tujuan Aplikasi:** <br> - Mengajarkan konsep dasar neural network <br> - Menunjukkan bagaimana model neural network dapat digunakan untuk masalah regresi linear.</p>", unsafe_allow_html=True)
        st.markdown("<p>**Cara Menggunakan:** <br> 1. Masukkan nilai untuk parameter a dan b dari fungsi linear. <br> 2. Masukkan nilai x untuk memprediksi y menggunakan fungsi linear yang ditentukan. <br> 3. Klik tombol 'Predict' untuk melihat hasil prediksi dan metrik pelatihan.</p>", unsafe_allow_html=True)
        st.markdown("<p>**Referensi:** <br> - TensorFlow Documentation: https://www.tensorflow.org/docs <br> - NumPy Documentation: https://numpy.org/doc/stable/ <br> - Matplotlib Documentation: https://matplotlib.org/stable/contents.html <br> - Streamlit Documentation : https://docs.streamlit.io</p>", unsafe_allow_html=True)
        
if __name__ == "__main__":
    main()
