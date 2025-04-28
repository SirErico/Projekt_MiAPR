# Implementacja planowania ruchu w 2D na mapie zajętości za pomocą algorytmu RRT z próbkowaniem sterowanym gradientem ograniczeń z sieci neuronowej
Projekt zaliczeniowy z przedmiotu Metody i algorytmy planowania ruchu 

## 🛠  OPIS

- wybranie gotowej / utworzenie własnej mapy zajętości 2D
- spróbkowanie mapy - utworzenie datasetu (in: współrzędne, out: wolna/zajęta)
- wytrenowanie prostej sieci neuronowej (MLP) w PyTorch lub Tensorflow
- przygotowanie metody odpytywania (inferencji) sieci z odczytem gradientu (din/dout) w punkcie
- integracja sieci z planerem RRT - wyuczona sieć (teoretycznie może zastąpić mapę przy sprawdzaniu zajętości w punkcie) powinna zwracać zerowy gradient dla miejsc daleko od granic obszarów wolnych/zajętych, natomiast niezerowy gradient w pobliżu przeszkód; gradient należy uwzględnić w funkcji próbkującej - wylosowane próbki w obszarze przeszkód można przesunąć wzdłuż gradientu do obszaru wolnego; można uwzględnić gradient w algorytmie również na inny sposób wg własnego pomysłu; w rezultacie należy uzyskać mniejszą liczbę iteracji niż w algorytmie bez gradientu
- wizualizacja planowania ścieżki w Rviz, sprawdzenie czasu planowania, długości i kształtu ścieżek"




[Algorytm RRT jug](https://put-jug.github.io/lab-miapr/Lab%206%20-%20Algorytmy%20poszukiwania%20%C5%9Bcie%C5%BCki%20pr%C3%B3bkuj%C4%85ce%20przestrze%C5%84%20poszukiwa%C5%84%20na%20przyk%C5%82adzie%20RRT%20(Rapidly-exploring%20Random%20Tree).html)
[Algorytm RRT random article](https://theclassytim.medium.com/robotic-path-planning-rrt-and-rrt-212319121378)	
