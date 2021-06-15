# Biblioteka python do automatycznej anotacji danych video
# Skrócona instrukcja obsługi biblioteki
## Przygotowanie do pracy z biblioteką
Zakładając, że pracujemy na systemie operacyjnym Linux, przygotowanie do pracy z biblioteką należy zacząć od sklonowania repozytorium do dowolnego miejsca na dysku. W następnym kroku należy sklonować repozytorium obsługiwanego detektora oraz zainstalować narzędzie labelme. Następnie należy zainstalować zależności wymagane do działania obu tych narzędzi. Ostatnim krokiem jest utworzenie powiązania symbolicznego wewnątrz katalogu biblioteki, do katalogu z detektorem (polecenie `ln`).
## Plik konfiguracyjny (opcjonalnie)
Następnym, opcjonalnym krokiem jest modyfikacja ustawień wewnątrz pliku konfiguracyjnego (`loop_options.yaml`). W przeciwnym wypadku uruchomione zostaną ustawienia domyślne. Poniżej wyjaśnione zostały kolejne opcje dostępne do modyfikacji:
* `detector_name` - typ detektora, wstępnie obsługiwany jedynie „ultralytics_yolo”, czyli ten wykorzystany w trakcie rozwijania biblioteki
* `detector_path` – ścieżka do katalogu z detektorem, w naszym przypadku wcześniej utworzone powiązanie symboliczne
* `train_epochs` – liczba epok w fazie trenowania detektora
* `test_batch` - rozmiar paczki danych wejściowych w podczas detekcji (wykorzystywane w fazach walidacji i detekcji)
* `proj_dir` - katalog projektowy, w którym zapisywane będą postępy pracy
* `mute_stdout` - wyciszenie standardowego wyjścia detektora
* `conf_ignore_threshold` - poziom pewności detekcji, poniżej którego detekcje nie będą zapisywane
* `validantion_samples_amount` - liczba próbek do wykorzystania w fazie walidacji
* `training_size_incrementation` - procentowa część całości zbioru bazowego, jaka jest pobierana w każdej iteracji fazy trenowania
* `whole_dataset_dir` - ścieżka do całości zbioru walidacyjnego
* `whole_dataset_images_dir` - ścieżka do anotacji całego zbioru danych
* `train_dataset_dir` - ścieżka do całości zbioru walidacyjnego
* `val_dataset_dir` - ścieżka do całości zbioru walidacyjnego
* `train_labels` - zostanie utworzona, ścieżka przechowująca tymczasowe, aktualne anotacje zbioru do trenowania
* `val_labels` - ścieżka do anotacji całości zbioru walidacyjnego
* `temp_val_labels` - zostanie utworzona, ścieżka przechowująca tymczasowe anotacje do małego zbioru walidacyjnego
* `temp_val_dataset_dir` - zostanie utworzona, ścieżka przechowujaca tymczasowe anotacje do małego zbioru walidacyjnego
* `dataset_classes_names` - lista nazw klas obiektów
## Wywołanie skryptu obsługującego bibliotekę
Interfejs biblioteki został zrealizowany poprzez wiersz poleceń. Skryptem wywoływanym jako główny interfejs w programie jest skrypt `loop.py` Wywołanie skryptu z opcją `--help` lub `--h` wyświetli dokładny sposób wywołania skryptu. Do wywołania skryptu potrzebne są 2 opcje. Pierwsza z nich to `--step`, czyli konkretna faza pętli Active Learningu do wykonania dostępne opcje to:
* training
* detection
* validation
* select_samples
* finish_iterations
Drugą opcją, czyli `--cfg` jest ścieżka do pliku konfiguracyjnego.
