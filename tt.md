1. Spłaszczone tabele finansowe (Wszystkie liczby w jednej kolumnie)

To obecnie najpoważniejszy problem w raportach finansowych. Skrypt generuje tabele Markdown, które mają strukturę dwukolumnową (|---|---|), mimo że oryginalnie zawierają one dane dla 4 różnych okresów. W efekcie wszystkie liczby zlewają się w jeden ciąg znaków rozdzielony spacjami.

    Przykład (XTB, Jednostkowe sprawozdanie z dochodów całkowitych):
    Markdown

    |**całkowitych**||
    |---|---|
    |**(w tys. PLN)**|**OKRES 3 MIESIĘCY ZAKOŃCZONY OKRES 9 MIESIĘCY ZAKOŃCZONY...**|
    |Wynik z operacji na instrumentach finansowych|296 209 1 306 985 421 599 1 248 988|

    Dlaczego to błąd? Liczby 296 209, 1 306 985, 421 599 i 1 248 988 znajdują się wewnątrz jednej komórki. Ponieważ w tabelach finansowych spacje służą jako separatory tysięczne, model LLM nie będzie w stanie poprawnie zmapować, gdzie kończy się jedna kwota, a zaczyna druga (np. czy 296 209 1 to jedna liczba, czy dwie). Każdy okres musi bezwzględnie posiadać własną kolumnę Markdown (np. | Wynik... | 296 209 | 1 306 985 | ... |).

2. Rozbite tabele tekstowe i iniekcja separatorów (Inne układy kolumn)

W tabelach opisowych (np. przy analizie ryzyk) skrypt gubi się na granicach stron i potrafi wstrzyknąć nagłówek lub separator tabeli wewnątrz wierszy z danymi, zmieniając przy tym losowo liczbę kolumn.

    Przykład (XTB, Istotne czynniki ryzyka):
    W środku opisu ryzyka nagle pojawia się linia separatora z zupełnie inną liczbą kolumn (najpierw 4, potem nagle 8):
    Markdown

    |...stabilność głównej platformy transakcyjnej ulokowanej w bezpiecznych strefach. plany i ćwiczenia BCP/DR.|...|
    |---|---|---|---|---|---|---|---|
    |czasowe, lokalne zakłócenia w obsłudze klienta...|...|

    Taki zapis całkowicie niszczy strukturę tabeli dla parsera Markdown.

3. Zdania urwane i rozcięte strukturą tabeli

Z powodu złego parsowania wierszy, zdania, które powinny stanowić jeden ciągły tekst wewnątrz komórki, zostają fizycznie rozcięte na dwa osobne wiersze tabeli.

    Przykład (XTB, Ryzyko fizyczne i zmian klimatu):
    Tekst w jednej komórce urywa się słowami:

        "...Zdarzenia te mogą powodować Spółka stosuje dywersyfikację geograficzną biur..."

    A dopiero w kolejnym wierszu tabeli (pod separatorem) pojawia się dokończenie myśli:

        "|czasowe, lokalne zakłócenia w obsłudze klienta i wzrost kosztów..."

    W efekcie zdanie logiczne brzmiące: "Zdarzenia te mogą powodować czasowe, lokalne zakłócenia..." zostało przerwane w połowie i przedzielone składnią tabeli.

4. Problem gubienia samogłosek (Dropped Vowels) – nadal występuje

Mimo poprawek, w niektórych sekcjach dokumentów (prawdopodobnie w specyficznych plikach PDF o uszkodzonym mapowaniu czcionek) tekst wciąż traci litery.

    Przykład (Fragment z losowego chunk-u tekstowego):
    W danych pojawiają się deformacje takie jak:

        "...zaichaa, w związku z tym zysk a akcje z działalści ktyuwaj jst rowy zyskwi a akcje..."

    Parser zniekształcił tutaj kluczowe pojęcia finansowe i gramatyczne: zaichaa (zaniechanej), działalści (działalności), ktyuwaj (kontynuowanej), jst (jest), rowy (równy), zyskwi (zyskowi), a akcje (na akcję).

Jak to ostatecznie rozwiązać w kodzie?

    Dedykowany ekstraktor tabel finansowych:
    Zamiast przetwarzać całą stronę PDF jako jeden wielki blok tekstu zamieniany na Markdown, musisz podejść do tabel modularnie. Wykorzystaj bibliotekę pdfplumber (funkcja .extract_tables()) lub wbudowany moduł wykrywania tabel w PyMuPDF (page.find_tables()). Wyciągaj tabele jako czyste macierze (listy list w Pythonie), usuwaj z nich zbędne spacje tysięczne wewnątrz liczb na czas formatowania i dopiero wtedy programistycznie buduj z nich poprawny Markdown z odpowiednią liczbą kolumn.

    Łączenie tekstu rozbitego na wiersze:
    Jeśli komórka w tabeli tekstowej kończy się małą literą lub spójnikiem (np. powodować), a następny wiersz zaczyna się od małej litery, Twój skrypt powinien automatycznie scalać te komórki w jeden tekst przed wstawieniem znaków | Markdownu.

    Problem z czcionkami (Dropped Vowels):
    Jeżeli standardowy silnik PDF (np. pdfminer / PyMuPDF) wypluwa słowa typu ktyuwaj, oznacza to wadę struktury ToUnicode w samym pliku PDF. Jedynym w 100% skutecznym rozwiązaniem dla takich stron jest zastosowanie warstwy OCR (np. Tesseract lub silnik EasyOCR/PaddleOCR), która przeczyta ten fragment jak obraz, zamiast pobierać uszkodzone mapowanie znaków bezpośrednio z pliku.
