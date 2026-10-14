# RINGS: история изменений моделей

Составлено 6 октября 2026 года по 42 Python-файлам из `rings_minigrid_brian_2_prototype.zip`, с сопоставлением кода и заголовков версий.

## Как читать историю

Номер файла и версия в заголовке иногда различаются. Ниже каждый файл учтён отдельно. Последовательность по номеру — порядок сравнения, а не доказанная история коммитов; дат создания версий архив не подтверждает. «По коду» обозначает интерпретацию назначения изменения. Заявленные улучшения поведения — цели разработчика, а не результаты статистической проверки.

В архиве нет отдельных файлов 7 и 25. Ветка `5_7_AB` не заменяет отсутствующую основную v0.07. В файле 37 находится сразу 0.37.2.

## Основные этапы

| Этап | Содержание |
|---|---|
| Начало — 5.2 | Локальная сенсорика, моторные кольца, перенос правил выбора действий в сеть, исправления среды |
| 6 — 18 | Curiosity, Fatigue, MotorRest и временная динамика Idle |
| 19 — 23 | Диагностика, слабые сенсорные сигналы, секторное восприятие |
| 24 — 32 | Движущиеся опасности, контактная угроза, голод и направленное избегание |
| 33 — 36 | Обзорные повороты, конкуренция моторных состояний, настройка реакций на еду и угрозы |
| 37 — 37.2 | Параллельные эксперименты, исправление пропущенной связи, прогресс и остановка процессов |

## История по файлам

### Начальный прототип — `rings_minigrid_brian_2_prototype.py`

**Что изменено.** Замкнутый цикл MiniGrid → сенсоры → спайковые кольца Explore, EnergyLow, FoodSeeking, Danger, Escape → действие. Визуализация среды, колец и активности; выбор действий содержит явные правила.

**Зачем.** Проверить принцип управления поведением через устойчивые состояния SNN. Это демонстрационный исходный уровень, а не оптимизированный агент.

### v0.02 — `rings_minigrid_brian_2_prototype2.py`

**Что изменено.** Введены RingsParams, случайный поиск sample_params/tune, загрузка параметров, EpisodeResult и корректный reset(). Оценка: mean_steps + 8×mean_food − 4×mean_danger.

**Зачем.** Автоматизировать подбор сети и получать численные результаты эпизодов; исправить интерфейс Gymnasium.

### Вариант v0.02 — `rings_minigrid_brian_2_prototype2.1.py`

**Что изменено.** Score заменён на 60×mean_food − 45×mean_danger + 0.20×mean_steps + 0.10×mean_reward. В tune отключена визуализация; строка visual переводится в bool через сравнение с on. Локальные rates переименованы.

**Зачем.** Явно заявленная цель нового score — гомеостатический успех, а не просто долгая жизнь. Исправление visual предотвращает трактовку непустой строки off как True. Причина переименования переменных отдельно не указана.

### v0.03 — `rings_minigrid_brian_2_prototype3.py`

**Что изменено.** Глобальные датчики ближайшего объекта заменены локальными эгоцентрическими лучами left/ahead/right; затухание сигнала с расстоянием; sensor_range стал параметром.

**Зачем.** Ограничить знание агента локально доступной сенсорной информацией.

### Вариант v0.03 — `rings_minigrid_brian_2_prototype3.1.py`

**Что изменено.** Добавлена отдельная панель сенсоров с численными значениями; окно расширено с трёх до четырёх панелей.

**Зачем.** По коду: сделать видимым вход сети и облегчить диагностику поведения.

### v0.04 — `rings_minigrid_brian_2_prototype4.py`

**Что изменено.** Добавлены MotorLeft, MotorRight, MotorForward; декодирование действий из моторных колец, подписи последнего действия и доминирующего состояния.

**Зачем.** Перенести выбор действий в нейронную динамику. В этой стадии в декодере ещё остаются защитные правила и смещение к еде.

### Вариант v0.04 — `rings_minigrid_brian_2_prototype4.1.py`

**Что изменено.** Food.can_overlap() возвращает True, can_pickup() — False.

**Зачем.** Позволить агенту входить на клетку еды и потреблять её: у исходного Ball проходимость не соответствовала этой механике.

### В файле указано v0.05 — `rings_minigrid_brian_2_prototype4.2.py`

**Что изменено.** Убрана жёсткая проверка опасности из декодера; food_forward_bias сделан параметром и установлен в 0; Danger явно проходим; исправлены строки заголовка.

**Зачем.** Проверять избегание опасности через SNN, а не через встроенный защитный алгоритм; стены остаются ограничением допустимости движения.

### v0.05 — `rings_minigrid_brian_2_prototype5.py`

**Что изменено.** Danger переведён с наследования Lava на красный Ball с семантическим type=danger; изменён заголовок визуализации.

**Зачем.** Избежать специальной логики Lava со стороны MiniGrid и контролировать ущерб собственной гомеостатической моделью.

### Вариант v0.05 — `rings_minigrid_brian_2_prototype5.1.py`

**Что изменено.** Перед super().step() сохраняются энергия и счётчики еды/контактов; затем они явно восстанавливаются и обновляются.

**Зачем.** В комментарии заявлено предотвращение нежелательного сброса гомеостатических переменных при шаге базовой среды. Наличие реального сброса в конкретной установленной библиотеке этим комментарием не доказано.

### Вариант v0.05 — `rings_minigrid_brian_2_prototype5.2.py`

**Что изменено.** Переписано формирование заголовка: энергия, еда, опасности, действие и состояние; food_forward_bias=0.0 и удалена закомментированная строка.

**Зачем.** По коду: улучшить читаемость и привести оформление к единообразию; отдельного нового механизма поведения нет.

### Отдельная ветка 5.7 / AB — `rings_minigrid_brian_2_prototype5_7_AB.py`

**Что изменено.** Экспорт Colab, упрощённые восемь групп нейронов и другой API. Food/Danger различаются флагами is_food/is_danger. Есть явные правила подачи моторных входов при еде, стенах и серии четырёх движений вперёд. Загрузка зависимостей, пошаговые графики; main вместо run_episode.

**Зачем.** По коду: отдельная демонстрационная/отладочная ветка. Это не подтверждённая промежуточная версия основной линии; авторская цель AB в файле не расшифрована.

### v0.06 — `rings_minigrid_brian_2_prototype6.py`

**Что изменено.** Добавлены кольца Fatigue и Curiosity, поперечные связи all-to-all. Fatigue получает активность Explore/FoodSeeking и тормозит их; Curiosity имеет внутреннее возбуждение и подавляется значимыми состояниями.

**Зачем.** Сформировать внутренние мотивационные и отрицательные обратные связи; уменьшить внешнее случайное исследование в декодере. Это ещё нейронные кольца, не отдельные химические модуляторы.

### v0.08 — `rings_minigrid_brian_2_prototype8.py`

**Что изменено.** Убрано резервное движение при молчании моторных колец. В доступном сравнении с v0.06 появляются MotorRest, связи усталости/отдыха и отдельная обработка action=6 со сниженной энергозатратой.

**Зачем.** Движение должно зависеть от моторной активности, а отдых — стать явным состоянием. Версия 7 отсутствует: нельзя уверенно распределить все изменения между 7 и 8.

### v0.09 — `rings_minigrid_brian_2_prototype9.py`

**Что изменено.** Добавлен Idle-вход; слабый путь Idle → Curiosity; Curiosity → Explore → моторика.

**Зачем.** Накапливать исследовательский импульс при отсутствии еды, угрозы и сигнала низкой энергии.

### v0.10 — `rings_minigrid_brian_2_prototype10.py`

**Что изменено.** Fatigue получает реальную моторную активность MotorForward/Left/Right; MotorRest её тормозит. Усилен Explore → MotorForward.

**Зачем.** Усталость должна отражать движение, а не намерение исследовать; отдых должен снижать усталость.

### v0.11 — `rings_minigrid_brian_2_prototype11.py`

**Что изменено.** Локальные датчики ограничены тремя клетками; квадратичное затухание 1.00/0.44/0.11. Замедлено накопление Fatigue.

**Зачем.** Создать больше периодов без значимых сигналов, чтобы проявлялся Curiosity; дать несколько шагов движения до отдыха. Указанные в комментарии 5–6 шагов — целевой замысел, не доказанный результат.

### v0.12 — `rings_minigrid_brian_2_prototype12.py`

**Что изменено.** Боковая опасность слабо возбуждает MotorForward; лучший набор сохраняется сразу; добавлен текстовый статус без визуализации; история ограничена 20 шагами.

**Зачем.** Уменьшить бесконечные повороты left/right при угрозе и сделать длительный tune наблюдаемым и менее уязвимым к потере результата.

### v0.13 — `rings_minigrid_brian_2_prototype13.py`

**Что изменено.** Настраиваемые Idle → Curiosity → Explore → Forward; прямой слабый Curiosity → Forward; реально используется w_explore_to_motor_forward. В score добавлен штраф 80×mean_rest_ratio. StateMonitor включается через record.

**Зачем.** Устранить постоянный отдых, не поощрять выживание без деятельности; сократить расход памяти при подборе.

### v0.14 — `rings_minigrid_brian_2_prototype14.py`

**Что изменено.** Добавлена дополнительная временная панель Idle/Curiosity/Explore и моторных состояний; правки строк.

**Зачем.** Наблюдать последовательность внутренних состояний и действий во времени.

### v0.15 — `rings_minigrid_brian_2_prototype15.py`

**Что изменено.** Временная панель заменена дорожками Sensors / State rings / Modulatory rings / Motor rings; исправлено оформление управляющих символов строк.

**Зачем.** Сделать причинную последовательность Idle → Curiosity → Explore → движение → Fatigue/Rest более читаемой.

### v0.16 — `rings_minigrid_brian_2_prototype16.py`

**Что изменено.** Убрана история активаций колец, расширена временная диаграмма. Бинарный Idle заменён накопительным idle_level; он сбрасывается при значимых сигналах, включая стену впереди.

**Зачем.** Отделить длительное бездействие от одного случайного тихого кадра; целевой порог накопления — около 5–6 шагов.

### v0.17 — `rings_minigrid_brian_2_prototype17.py`

**Что изменено.** idle_level обновляется один раз за управляющий шаг вне encode_observation(); усилено сохранение Curiosity, увеличена его постоянная времени и ослаблено торможение.

**Зачем.** Устранить побочные эффекты повторного чтения наблюдения; сделать Curiosity состоянием на несколько шагов.

### v0.18 — `rings_minigrid_brian_2_prototype18.py`

**Что изменено.** Стена исключена из условия Idle. При Explore/Curiosity у стены декодер смещает действие к повороту; движение в стену по-прежнему запрещено.

**Зачем.** Не позволять агенту навсегда застывать лицом к стене. Часть моторной коррекции остаётся явным правилом декодера.

### v0.19 — `rings_minigrid_brian_2_prototype19.py`

**Что изменено.** Добавлен CSV --debug-log с условиями Idle, сенсорами, активациями, действиями и событиями.

**Зачем.** Диагностировать, почему idle_level растёт или не растёт.

### v0.20 — `rings_minigrid_brian_2_prototype20.py`

**Что изменено.** Добавлены food_motor_threshold, отдельные idle_food_threshold/idle_danger_threshold и реакция на слабую еду впереди.

**Зачем.** Слабый удалённый сигнал еды должен вызывать движение, а не блокировать Idle при отсутствии действия. Следующая версия отдельно устраняет декодерное усиление этого этапа.

### v0.21 — `rings_minigrid_brian_2_prototype21.py`

**Что изменено.** Отдельные SNN-пути food sensors → motor rings; усилен lateral danger → Forward; убрано декодерное усиление food_motor_threshold.

**Зачем.** Получать ориентацию на слабую боковую еду и уход от угрозы через динамику сети.

### v0.22 — `rings_minigrid_brian_2_prototype22.py`

**Что изменено.** Убрано оставшееся условие еды в декодере; добавлен Escape → Forward; danger_ahead симметрично возбуждает оба поворота.

**Зачем.** Завершить перенос управления едой в SNN и обеспечить возможность движения при escape.

### v0.23 — `rings_minigrid_brian_2_prototype23.py`

**Что изменено.** Лучевые датчики заменены секторными: диагональный объект влияет на передний и боковой каналы. Пересмотрено энергетическое условие Idle.

**Зачем.** Воспринимать диагональные объекты и позволять Curiosity действовать при умеренном уровне энергии.

### v0.24 — `rings_minigrid_brian_2_prototype24.py`

**Что изменено.** Размер мира по умолчанию 15×15; число еды/опасностей сохранено. Добавлены grid_size и danger_speed; опасности случайно перемещаются, а рядом преследуют агента, в том числе по диагонали.

**Зачем.** Проверять поведение в более просторном мире и при активной угрозе. Скорость 0 сохраняет статические опасности.

### v0.26 — `rings_minigrid_brian_2_prototype26.py`

**Что изменено.** Движение опасности ограничено ортогональными шагами; при диагональном соседстве выбирается один из двух сближающих шагов. По сравнению с 24 запрещён заход на любую занятую клетку, включая еду.

**Зачем.** Сделать подвижность угрозы сопоставимой с агентом и сохранить еду. Заголовок относит сохранение еды к отсутствующей v0.25; её самостоятельный код не проверен.

### v0.27 — `rings_minigrid_brian_2_prototype27.py`

**Что изменено.** Усилены Danger/Escape → Forward; Danger/Escape тормозят MotorRest и Fatigue; версия показана в окне.

**Зачем.** Не допускать отдыха во время атаки и ускорить выход из опасности.

### v0.28 — `rings_minigrid_brian_2_prototype28.py`

**Что изменено.** Добавлен danger_contact — отдельный вход для опасности на клетке агента; возбуждает Danger/Escape/Forward. Контакт отображён в сенсорах, угрозы выделены красным.

**Зачем.** Закрыть слепую зону направленных сенсоров, когда опасность уже совпадает с позицией агента.

### v0.29 — `rings_minigrid_brian_2_prototype29.py`

**Что изменено.** Прямой путь danger sensors → Escape, усиление боковой угрозы → Forward, danger_input_gain.

**Зачем.** Надёжнее запускать защитное движение при видимой, но ещё неконтактной угрозе вместо отдыха.

### v0.30 — `rings_minigrid_brian_2_prototype30.py`

**Что изменено.** Усилены EnergyLow → Explore и FoodSeeking → Forward; голод/поиск тормозят MotorRest.

**Зачем.** Не позволять агенту умирать в покое при низкой энергии и отсутствии видимой еды.

### v0.31 — `rings_minigrid_brian_2_prototype31.py`

**Что изменено.** danger_ahead тормозит Forward и сильнее возбуждает повороты; убран из пути опасность → движение вперёд.

**Зачем.** Не бежать прямо в угрозу, находящуюся перед агентом.

### v0.32 — `rings_minigrid_brian_2_prototype32.py`

**Что изменено.** danger_left тормозит MotorLeft, danger_right тормозит MotorRight.

**Зачем.** При уходе от стены или угрозы не поворачивать в сторону видимой боковой опасности.

### v0.33 — `rings_minigrid_brian_2_prototype33.py`

**Что изменено.** AGENT_VERSION; agent_view_size согласован с радиусом сенсоров; CSV по trial/seed в tune; Idle/Curiosity слабо возбуждают обзорные повороты.

**Зачем.** Показывать восприятие согласованно и позволять обнаруживать приближающуюся сзади опасность за счёт смены ориентации.

### v0.34 — `rings_minigrid_brian_2_prototype34.py`

**Что изменено.** Создан настраиваемый S_escape_turn: Escape → Left/Right; добавлена слабая самоподдержка моторных колец; ослаблено доминирование Rest. ВАЖНО: S_escape_turn отсутствует в явном Network и фактически не исполняется.

**Зачем.** Замысел — конкуренция поворотов при escape и более устойчивые действия. Для пути Escape → повороты замысел не реализован до 0.37.1.

### v0.35 — `rings_minigrid_brian_2_prototype35.py`

**Что изменено.** Forward побеждает повороты только при заметном превосходстве; усилены боковые пищевые пути к соответствующим поворотам. Пропуск S_escape_turn сохраняется.

**Зачем.** Уменьшить предпочтение движения прямо, когда еда расположена сбоку.

### v0.36 — `rings_minigrid_brian_2_prototype36.py`

**Что изменено.** Усилены торможение Forward при danger_ahead и возбуждение поворотов; ослаблены общий Escape → Forward и моторная самоподдержка. Контактный и боковой уход вперёд сохранены. S_escape_turn всё ещё не включён.

**Зачем.** Разделить реакции: передняя угроза требует поворота, боковая/контактная может требовать рывка вперёд.

### v0.37.2 в архиве — `rings_minigrid_brian_2_prototype37.py`

**Что изменено.** Параллельный tune (1–30 процессов), общий набор seed, checkpoint/resume; кодирование Food/Danger для MiniGrid. Добавлен ранее пропущенный S_escape_turn; живой прогресс. Остановка пула при Ctrl-C/SIGTERM, узкий фильтр warning distutils, печать версии Python.

**Зачем.** Ускорить подбор, сделать длительные серии наблюдаемыми и прерываемыми; исправить реальную потерю связи в модели. Подэтапы 0.37/0.37.1/0.37.2 описаны отдельно ниже.

## Детализация серии 0.37

| Подверсия | Изменения и причина |
|---|---|
| 0.37 | Параллельный поиск кандидатов, настраиваемые сценарии, атомарное сохранение и resume; устранение ошибки кодирования объектов MiniGrid. Цель — использовать 30 ядер и расширить оценку. |
| 0.37.1 | S_escape_turn включён в Network; добавлен прогресс внутри эпизодов. Исправлен warning, за которым стояла неработающая связь, и длительное молчание tune. |
| 0.37.2 | Пул явно terminate/join при Ctrl-C/SIGTERM, точечное подавление предупреждения distutils, версия/путь Python при старте. Цель — не оставлять рабочих процессов после остановки. |

Разбиение 0.37 на подэтапы подтверждено нашей перепиской и итоговым кодом; отдельных снимков 0.37 и 0.37.1 в этом архиве нет.

## Что важно при интерпретации результатов

1. **Критерий tune менялся.** В файле 2: средние шаги + 8×еда − 4×контакты. В 2.1: 60×еда − 45×контакты + 0.20×шаги + 0.10×награда. В 13 добавлен −80×средняя доля отдыха. Повторяющаяся фраза «first optimization criterion is survival time» в заголовках поздних файлов устарела относительно кода.
2. **Менялись не только сети, но и условия задачи.** Проходимость еды/опасностей, отдых, сенсоры, размер мира и движения угроз влияют на результат. Одинаковый seed сам по себе не делает старые среды эквивалентными.
3. **S_escape_turn в 34–36 был только создан.** Наличие объекта Synapses и его параметра ещё не означает участия в явном Network. Исправление 0.37.1 меняет поведение, поэтому 37.2 — не только ускоренная 36.
4. **Нейромодуляторная архитектура из обсуждения ещё не внедрена.** Fatigue и Curiosity в основной линии остаются кольцами. Специальные выделяющие нейроны, концентрации с экспоненциальным распадом и ForagingMod/DefenseMod — будущий проект, не результат этих версий.
5. **Совместимость benchmark.** Без адаптации интерфейса подходят файлы 24 и 26–37 с одинаково включённым `--minigrid-compat`. Проверены 26 коротких эпизодов: по два на версию, по пять шагов. Это проверка исполнения, не качества.
6. **Группы одинакового кода класса среды:** 26–27; 28–32; 33–37. Класс среды у 24 отдельный. Совпадение класса не гарантирует совпадение всех фактических условий: проверяйте передаваемые параметры, сенсорный радиус и классы объектов.

## Техническое приложение: проверяемые различия кода

Ниже — автоматическое сравнение с предыдущим доступным файлом основной последовательности. Игнорируются комментарии и docstring; перечисляются изменённые функции/методы и изменения значений RingsParams по умолчанию. Это помогает увидеть правки, не вошедшие в авторское описание. Ветка AB сравнивается с 5.2 отдельно; версия 6 также сравнивается с 5.2. Изменения диапазонов случайного поиска отражаются наличием sample_params среди изменённых функций.

### Файл `rings_minigrid_brian_2_prototype.py`

SHA-256: `3f702c2d25485df7034a859b5ab2f546dc1938027471726a883daa04409ea0c5`

### Файл `rings_minigrid_brian_2_prototype2.py`

SHA-256: `5846632e2795eec48cc74a1faf92f506e10a2753be4cc022faedbcef87e36a19`

Сравнение с `rings_minigrid_brian_2_prototype.py`.

Функции: `Δ RingsHomeostasisEnv.__init__`; `Δ RingsHomeostasisEnv._gen_grid`; `Δ RingsHomeostasisEnv._wall_ahead`; `+ RingsHomeostasisEnv.reset`; `− RingsHomeostasisEnv.reset2`; `Δ RingsHomeostasisEnv.step`; `Δ RingsSNNController.__init__`; `Δ RingsSNNController._connect_ring_to_ring`; `Δ RingsSNNController.decode_action`; `Δ RingsSNNController.encode_observation`; `Δ RingsSNNController.step`; `Δ RingsVisualizer.update`; `+ evaluate_params`; `+ load_params`; `+ main`; `Δ run_episode`; `+ sample_params`; `+ tune_random_search`.

Параметры по умолчанию:

- `danger_threshold`: `отсутствует` → `0.12`
- `energy_low_threshold`: `отсутствует` → `0.55`
- `escape_activation_threshold`: `отсутствует` → `0.05`
- `food_threshold`: `отсутствует` → `0.05`
- `input_rate_hz`: `отсутствует` → `220.0`
- `ring_size`: `отсутствует` → `12`
- `seeking_activation_threshold`: `отсутствует` → `0.02`
- `snn_step_ms`: `отсутствует` → `30`
- `w_cross_exc`: `отсутствует` → `4.5`
- `w_cross_inh`: `отсутствует` → `5.5`
- `w_input`: `отсутствует` → `8.0`
- `w_ring`: `отсутствует` → `7.0`
- `w_support`: `отсутствует` → `2.2`

### Файл `rings_minigrid_brian_2_prototype2.1.py`

SHA-256: `4325c11eed44c788b4c065b30b2b8cc1507248b28a08c4db81c0242413632fc7`

Сравнение с `rings_minigrid_brian_2_prototype2.py`.

Функции: `Δ RingsSNNController.decode_action`; `Δ RingsSNNController.step`; `Δ evaluate_params`; `Δ main`; `Δ tune_random_search`.

### Файл `rings_minigrid_brian_2_prototype3.py`

SHA-256: `d899986dc377e68d70d80167a020f461c2db035988f39f57a62a790568bf8091`

Сравнение с `rings_minigrid_brian_2_prototype2.1.py`.

Функции: `Δ RingsHomeostasisEnv.__init__`; `+ RingsHomeostasisEnv._dir_vec`; `Δ RingsHomeostasisEnv._get_obs`; `− RingsHomeostasisEnv._nearest_obj_vector`; `+ RingsHomeostasisEnv._ray_sensor`; `+ RingsHomeostasisEnv._relative_dirs`; `Δ RingsSNNController.encode_observation`; `Δ run_episode`; `Δ sample_params`.

Параметры по умолчанию:

- `sensor_range`: `отсутствует` → `5`

### Файл `rings_minigrid_brian_2_prototype3.1.py`

SHA-256: `34fe790bd036233cd8bf57caff1b4eba1286ede33a9bab236337b2cb01c5dbc3`

Сравнение с `rings_minigrid_brian_2_prototype3.py`.

Функции: `Δ RingsVisualizer.__init__`; `Δ RingsVisualizer.update`.

### Файл `rings_minigrid_brian_2_prototype4.py`

SHA-256: `4674be4eed2b05721c18af42048b9fc5f8a9c3a81cf85c0132b67e331a74ef9d`

Сравнение с `rings_minigrid_brian_2_prototype3.1.py`.

Функции: `Δ RingsHomeostasisEnv._relative_dirs`; `Δ RingsHomeostasisEnv.reset`; `Δ RingsHomeostasisEnv.step`; `Δ RingsSNNController.__init__`; `Δ RingsSNNController.decode_action`; `Δ RingsVisualizer.update`.

### Файл `rings_minigrid_brian_2_prototype4.1.py`

SHA-256: `10f7c2db6348b6b2b77a50118e95cb5d6dbf291c8164c312579f734d0249a932`

Сравнение с `rings_minigrid_brian_2_prototype4.py`.

Функции: `+ Food.can_overlap`; `+ Food.can_pickup`.

### Файл `rings_minigrid_brian_2_prototype4.2.py`

SHA-256: `d72154d85b177c50bca4016918157f862baa8b8eece4ad015f5c08a600481934`

Сравнение с `rings_minigrid_brian_2_prototype4.1.py`.

Функции: `+ Danger.can_overlap`; `+ Danger.can_pickup`; `Δ RingsSNNController.decode_action`; `Δ RingsVisualizer.update`; `Δ sample_params`.

Параметры по умолчанию:

- `food_forward_bias`: `отсутствует` → `0.0`

### Файл `rings_minigrid_brian_2_prototype5.py`

SHA-256: `b68f02a9efd03d7a383169ca5fdf462465d8d38cb3840aa3850f026fad4bf952`

Сравнение с `rings_minigrid_brian_2_prototype4.2.py`.

Функции: `Δ Danger.__init__`; `Δ RingsVisualizer.update`.

### Файл `rings_minigrid_brian_2_prototype5.1.py`

SHA-256: `3786cd2452a110559d7b130069889eb5642cc5b09bf87181daa68ef2ab377339`

Сравнение с `rings_minigrid_brian_2_prototype5.py`.

Функции: `Δ RingsHomeostasisEnv.step`; `Δ RingsVisualizer.update`.

### Файл `rings_minigrid_brian_2_prototype5.2.py`

SHA-256: `f4efb1b9726d2f9ec98ab252a1b0713403a425b86434a637da3db4c7b603d5b9`

Сравнение с `rings_minigrid_brian_2_prototype5.1.py`.

Функции: `Δ RingsVisualizer.update`; `Δ sample_params`.

### Файл `rings_minigrid_brian_2_prototype5_7_AB.py`

SHA-256: `f09d78c54124b2fe6bbe9831be0b4b378b8490122fd4696ea1c16017224345f3`

Сравнение с `rings_minigrid_brian_2_prototype5.2.py`.

Функции: `Δ Danger.__init__`; `Δ Danger.can_overlap`; `− Danger.can_pickup`; `Δ Food.__init__`; `Δ Food.can_overlap`; `− Food.can_pickup`; `Δ RingsHomeostasisEnv.__init__`; `− RingsHomeostasisEnv._dir_vec`; `Δ RingsHomeostasisEnv._gen_grid`; `Δ RingsHomeostasisEnv._get_obs`; `− RingsHomeostasisEnv._ray_sensor`; `− RingsHomeostasisEnv._relative_dirs`; `− RingsHomeostasisEnv._wall_ahead`; `+ RingsHomeostasisEnv.gen_obs_grid`; `Δ RingsHomeostasisEnv.reset`; `Δ RingsHomeostasisEnv.step`; `+ RingsParams.from_json`; `Δ RingsSNNController.__init__`; `− RingsSNNController._connect_input_to_ring`; `− RingsSNNController._connect_ring_to_ring`; `− RingsSNNController._ring_idx`; `− RingsSNNController.decode_action`; `− RingsSNNController.encode_observation`; `− RingsSNNController.ring_activations`; `Δ RingsSNNController.step`; `− RingsVisualizer.__init__`; `− RingsVisualizer.update`; `− evaluate_params`; `+ install_if_missing`; `− load_params`; `Δ main`; `− run_episode`; `− sample_params`; `− tune_random_search`.

Параметры по умолчанию:

- `danger_threshold`: `0.12` → `удалён`
- `energy_low_threshold`: `0.55` → `0.5722`
- `escape_activation_threshold`: `0.05` → `удалён`
- `food_forward_bias`: `0.0` → `удалён`
- `food_threshold`: `0.05` → `удалён`
- `input_rate_hz`: `220.0` → `175.3834`
- `ring_size`: `12` → `10`
- `seeking_activation_threshold`: `0.02` → `удалён`
- `sensor_range`: `5` → `3`
- `snn_step_ms`: `30` → `20`
- `w_cross_exc`: `4.5` → `удалён`
- `w_cross_inh`: `5.5` → `удалён`
- `w_input`: `8.0` → `8.1468`
- `w_ring`: `7.0` → `7.6042`
- `w_support`: `2.2` → `удалён`

### Файл `rings_minigrid_brian_2_prototype6.py`

SHA-256: `96fbafeab685b2acf3872cb3d5c788d553611ff9e0e28ed0709c8af1c8d694bb`

Сравнение с `rings_minigrid_brian_2_prototype5.2.py`.

Функции: `Δ RingsSNNController.__init__`; `+ RingsSNNController._connect_ring_to_ring_all`; `Δ RingsSNNController.decode_action`; `Δ RingsVisualizer.update`; `Δ sample_params`.

Параметры по умолчанию:

- `curiosity_bias_mv`: `отсутствует` → `-64.0`
- `curiosity_tau_ms`: `отсутствует` → `15.0`
- `enable_modulatory_rings`: `отсутствует` → `True`
- `fatigue_tau_ms`: `отсутствует` → `80.0`
- `w_curiosity_inh`: `отсутствует` → `0.9`
- `w_curiosity_to_explore`: `отсутствует` → `2.0`
- `w_fatigue_in`: `отсутствует` → `0.8`
- `w_fatigue_out`: `отсутствует` → `1.2`

### Файл `rings_minigrid_brian_2_prototype8.py`

SHA-256: `a88f5b76bafb190b5c8d1b98680ab1e28dd24d531a42e502dfff4933805fe330`

Сравнение с `rings_minigrid_brian_2_prototype6.py`.

Функции: `Δ RingsHomeostasisEnv.step`; `Δ RingsSNNController.__init__`; `Δ RingsSNNController.decode_action`; `Δ RingsVisualizer.update`; `Δ sample_params`.

Параметры по умолчанию:

- `fatigue_rest_threshold`: `отсутствует` → `0.05`
- `w_fatigue_motor_inh`: `отсутствует` → `2.0`
- `w_fatigue_to_rest`: `отсутствует` → `3.0`

### Файл `rings_minigrid_brian_2_prototype9.py`

SHA-256: `eef37ead48b1b6ced79661778acb283c8d8d310bb95e225771cfe843e7e6c402`

Сравнение с `rings_minigrid_brian_2_prototype8.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsSNNController.decode_action`; `Δ RingsSNNController.encode_observation`; `Δ sample_params`.

Параметры по умолчанию:

- `curiosity_bias_mv`: `-64.0` → `-65.5`
- `curiosity_tau_ms`: `15.0` → `90.0`
- `w_idle_to_curiosity`: `отсутствует` → `2.2`

### Файл `rings_minigrid_brian_2_prototype10.py`

SHA-256: `7c2a9c65150c1994bee72b138534378cf4b80a397bfb8841f3ec1afa5ffb63cf`

Сравнение с `rings_minigrid_brian_2_prototype9.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ sample_params`.

Параметры по умолчанию:

- `w_explore_to_motor_forward`: `отсутствует` → `3.0`
- `w_motor_to_fatigue`: `отсутствует` → `1.0`
- `w_rest_to_fatigue_inh`: `отсутствует` → `2.0`

### Файл `rings_minigrid_brian_2_prototype11.py`

SHA-256: `3b8e2f76130dde6e240c118fcdb96b2f070f648197168f6d9acd2778507e1deb`

Сравнение с `rings_minigrid_brian_2_prototype10.py`.

Функции: `Δ RingsHomeostasisEnv.__init__`; `Δ RingsHomeostasisEnv._ray_sensor`; `Δ sample_params`.

Параметры по умолчанию:

- `fatigue_rest_threshold`: `0.05` → `0.18`
- `fatigue_tau_ms`: `80.0` → `300.0`
- `sensor_range`: `5` → `3`
- `w_fatigue_motor_inh`: `2.0` → `1.2`
- `w_fatigue_to_rest`: `3.0` → `2.0`
- `w_motor_to_fatigue`: `1.0` → `0.25`

### Файл `rings_minigrid_brian_2_prototype12.py`

SHA-256: `eb0fa7577298e7c36772a52de2dc420a36fc264a2e3fdf13a9fa5e5ee7722dc4`

Сравнение с `rings_minigrid_brian_2_prototype11.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsVisualizer.update`; `Δ evaluate_params`; `Δ main`; `Δ run_episode`; `Δ sample_params`; `Δ tune_random_search`.

Параметры по умолчанию:

- `w_lateral_danger_to_forward`: `отсутствует` → `1.5`

### Файл `rings_minigrid_brian_2_prototype13.py`

SHA-256: `42343584ded367d025e949efecc5e55aefe554036c84ec58c2a859960aa93f7f`

Сравнение с `rings_minigrid_brian_2_prototype12.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ evaluate_params`; `Δ main`; `Δ run_episode`; `Δ sample_params`; `Δ tune_random_search`.

Параметры по умолчанию:

- `w_curiosity_to_motor_forward`: `отсутствует` → `1.2`
- `w_lateral_danger_to_forward`: `1.5` → `3.5`

### Файл `rings_minigrid_brian_2_prototype14.py`

SHA-256: `bbd664079f3f54c6dc2c78eafaebd9c5213daaf0b01bee5e3e4622821f6c7e20`

Сравнение с `rings_minigrid_brian_2_prototype13.py`.

Функции: `Δ RingsVisualizer.__init__`; `Δ RingsVisualizer.update`.

### Файл `rings_minigrid_brian_2_prototype15.py`

SHA-256: `8951765af8012af0c9e89cb64a96fa0d8cc8ec737880e1d19b4424c2418a969d`

Сравнение с `rings_minigrid_brian_2_prototype14.py`.

Функции: `Δ RingsVisualizer.update`; `Δ evaluate_params`.

### Файл `rings_minigrid_brian_2_prototype16.py`

SHA-256: `d9b49c17bf29ba847d8cf46b3421cf062879b2620578438a89818fff1cb3149d`

Сравнение с `rings_minigrid_brian_2_prototype15.py`.

Функции: `Δ RingsHomeostasisEnv.__init__`; `Δ RingsHomeostasisEnv._gen_grid`; `Δ RingsSNNController.encode_observation`; `Δ RingsVisualizer.__init__`; `Δ RingsVisualizer.update`; `Δ sample_params`.

Параметры по умолчанию:

- `idle_activation_threshold`: `отсутствует` → `0.8`
- `idle_decay_per_step`: `отсутствует` → `0.5`
- `idle_rise_per_step`: `отсутствует` → `0.2`

### Файл `rings_minigrid_brian_2_prototype17.py`

SHA-256: `f9921a8ed5ea9aeee7b5418506d26516ef58f5154e364a6c253c4f39081a66b6`

Сравнение с `rings_minigrid_brian_2_prototype16.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsSNNController.encode_observation`; `Δ RingsSNNController.step`; `+ RingsSNNController.update_idle_level`; `Δ sample_params`.

Параметры по умолчанию:

- `curiosity_bias_mv`: `-65.5` → `-65.2`
- `curiosity_tau_ms`: `90.0` → `220.0`
- `w_curiosity_inh`: `0.9` → `0.45`
- `w_curiosity_to_explore`: `2.0` → `2.6`

### Файл `rings_minigrid_brian_2_prototype18.py`

SHA-256: `f2bec0d06d4dc88f7e0f0a6dfca735b198decc80ac8218e1fd2e57d30129dc6f`

Сравнение с `rings_minigrid_brian_2_prototype17.py`.

Функции: `Δ RingsSNNController.decode_action`; `Δ RingsSNNController.update_idle_level`.

### Файл `rings_minigrid_brian_2_prototype19.py`

SHA-256: `a64cff3c70af3273ca6a6ca00ebb8f764e1249c0b77d40f5b20e3c34cc79b3fa`

Сравнение с `rings_minigrid_brian_2_prototype18.py`.

Функции: `Δ main`; `Δ run_episode`.

### Файл `rings_minigrid_brian_2_prototype20.py`

SHA-256: `d9b9d310ab56c013f1a22a1697cc1da06429daaa883a77e2f0a6b6e2fe60bbd1`

Сравнение с `rings_minigrid_brian_2_prototype19.py`.

Функции: `Δ RingsSNNController.decode_action`; `Δ RingsSNNController.update_idle_level`; `Δ sample_params`.

Параметры по умолчанию:

- `food_motor_threshold`: `отсутствует` → `0.1`
- `idle_danger_threshold`: `отсутствует` → `0.1`
- `idle_food_threshold`: `отсутствует` → `0.1`

### Файл `rings_minigrid_brian_2_prototype21.py`

SHA-256: `f8fac45bf1be51e0199dc99c2a7590caaf9c765a321c099b215d3701cfb9a6dc`

Сравнение с `rings_minigrid_brian_2_prototype20.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsSNNController.decode_action`; `Δ sample_params`.

Параметры по умолчанию:

- `w_food_sensor_to_motor`: `отсутствует` → `16.0`
- `w_lateral_danger_to_forward`: `3.5` → `4.5`

### Файл `rings_minigrid_brian_2_prototype22.py`

SHA-256: `9829cf0cdf6cb7e30ba0369b55ff87b23f897c8b6e172ac0ca7d754099101037`

Сравнение с `rings_minigrid_brian_2_prototype21.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsSNNController.decode_action`; `Δ sample_params`.

Параметры по умолчанию:

- `w_danger_ahead_to_turn`: `отсутствует` → `4.0`
- `w_escape_to_motor_forward`: `отсутствует` → `3.5`

### Файл `rings_minigrid_brian_2_prototype23.py`

SHA-256: `2f65e70e483183ed6f50c80baf1eafd7c0b3ceef2d14d87a0e8ef18b0c020d4b`

Сравнение с `rings_minigrid_brian_2_prototype22.py`.

Функции: `Δ RingsHomeostasisEnv._get_obs`; `− RingsHomeostasisEnv._ray_sensor`; `+ RingsHomeostasisEnv._sector_sensors`; `Δ RingsSNNController.update_idle_level`; `Δ RingsVisualizer.update`; `Δ run_episode`; `Δ sample_params`.

Параметры по умолчанию:

- `idle_energy_threshold`: `отсутствует` → `0.35`

### Файл `rings_minigrid_brian_2_prototype24.py`

SHA-256: `bbac44a5678a56aa7916372a71b5160d9e0ddc69fd8e1af165954d66fde103c8`

Сравнение с `rings_minigrid_brian_2_prototype23.py`.

Функции: `Δ RingsHomeostasisEnv.__init__`; `+ RingsHomeostasisEnv._danger_positions`; `Δ RingsHomeostasisEnv._gen_grid`; `+ RingsHomeostasisEnv._move_dangers_by_speed`; `+ RingsHomeostasisEnv._move_dangers_once`; `+ RingsHomeostasisEnv._sign`; `Δ RingsHomeostasisEnv.step`; `Δ evaluate_params`; `Δ main`; `Δ run_episode`; `Δ tune_random_search`.

### Файл `rings_minigrid_brian_2_prototype26.py`

SHA-256: `9b130cd0e25f198234e9b407d19b17503310402bfcdabfc9b36504df13cb72aa`

Сравнение с `rings_minigrid_brian_2_prototype24.py`.

Функции: `Δ RingsHomeostasisEnv._move_dangers_once`.

### Файл `rings_minigrid_brian_2_prototype27.py`

SHA-256: `d7d27cb02c2c3136d547c12717365cde124a177e20f3c0bf18fc924541d1eb7f`

Сравнение с `rings_minigrid_brian_2_prototype26.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsVisualizer.update`; `Δ sample_params`.

Параметры по умолчанию:

- `w_danger_escape_fatigue_inh`: `отсутствует` → `4.0`
- `w_danger_escape_rest_inh`: `отсутствует` → `4.0`
- `w_danger_to_motor_forward`: `отсутствует` → `5.0`
- `w_escape_to_motor_forward`: `3.5` → `7.0`

### Файл `rings_minigrid_brian_2_prototype28.py`

SHA-256: `0171da9cd2c2c063dfd4b94351d72c7d573584e5f25a4f668095211b9f2cc54e`

Сравнение с `rings_minigrid_brian_2_prototype27.py`.

Функции: `Δ RingsHomeostasisEnv.__init__`; `Δ RingsHomeostasisEnv._gen_grid`; `Δ RingsHomeostasisEnv._move_dangers_once`; `Δ RingsHomeostasisEnv.step`; `Δ RingsSNNController.__init__`; `Δ RingsSNNController.decode_action`; `Δ RingsSNNController.encode_observation`; `Δ RingsSNNController.update_idle_level`; `Δ RingsVisualizer.update`.

### Файл `rings_minigrid_brian_2_prototype29.py`

SHA-256: `8013df5a1bb509844527a9f55c64c9947872eb885bebfa78cb828bd156325328`

Сравнение с `rings_minigrid_brian_2_prototype28.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsSNNController.encode_observation`; `Δ sample_params`.

Параметры по умолчанию:

- `danger_input_gain`: `отсутствует` → `1.8`
- `w_danger_sensor_to_escape`: `отсутствует` → `14.0`
- `w_lateral_danger_to_forward`: `4.5` → `7.0`

### Файл `rings_minigrid_brian_2_prototype30.py`

SHA-256: `8223b0eaf8d0aa9fdb22934bf5b42d26dacd2effe807437c9f033242a3e0ae5b`

Сравнение с `rings_minigrid_brian_2_prototype29.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ sample_params`.

Параметры по умолчанию:

- `w_energy_low_to_explore`: `отсутствует` → `5.0`
- `w_foodseeking_to_motor_forward`: `отсутствует` → `7.0`
- `w_hunger_rest_inh`: `отсутствует` → `5.0`

### Файл `rings_minigrid_brian_2_prototype31.py`

SHA-256: `dbe311223941c69fb98e2d570370d73769e89df14a1d5eda1753b72acaddc091`

Сравнение с `rings_minigrid_brian_2_prototype30.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ sample_params`.

Параметры по умолчанию:

- `w_danger_ahead_forward_inh`: `отсутствует` → `10.0`
- `w_danger_ahead_to_turn`: `4.0` → `8.0`

### Файл `rings_minigrid_brian_2_prototype32.py`

SHA-256: `031661aedb7bd133ee6f29fc7506b8592196ae64bc612a794e9f84d20377e7c2`

Сравнение с `rings_minigrid_brian_2_prototype31.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsVisualizer.update`; `Δ sample_params`.

Параметры по умолчанию:

- `w_lateral_danger_turn_inh`: `отсутствует` → `10.0`

### Файл `rings_minigrid_brian_2_prototype33.py`

SHA-256: `bd47015b034c0f37fb7a0bc94a75b4f5e10204e8d904307179d86647dbdbbdc0`

Сравнение с `rings_minigrid_brian_2_prototype32.py`.

Функции: `Δ RingsHomeostasisEnv.__init__`; `Δ RingsSNNController.__init__`; `Δ RingsVisualizer.update`; `Δ evaluate_params`; `Δ main`; `Δ sample_params`; `Δ tune_random_search`.

Параметры по умолчанию:

- `w_idle_scan_turn`: `отсутствует` → `2.5`

### Файл `rings_minigrid_brian_2_prototype34.py`

SHA-256: `08ef8cc68cb94c040614f64b154c9838dd649e16edbed535ea9dad057e839445`

Сравнение с `rings_minigrid_brian_2_prototype33.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ sample_params`.

Параметры по умолчанию:

- `w_escape_to_turn`: `отсутствует` → `5.0`
- `w_hunger_rest_inh`: `5.0` → `6.0`
- `w_motor_self_support`: `отсутствует` → `1.8`

### Файл `rings_minigrid_brian_2_prototype35.py`

SHA-256: `19e55b9ee5e17958353debe467056a8b3b90dd0f28da2d287aa9837eea94f995`

Сравнение с `rings_minigrid_brian_2_prototype34.py`.

Функции: `Δ RingsSNNController.__init__`; `Δ RingsSNNController.decode_action`; `Δ sample_params`.

Параметры по умолчанию:

- `forward_win_margin`: `отсутствует` → `0.2`
- `w_lateral_food_sensor_to_motor`: `отсутствует` → `24.0`

### Файл `rings_minigrid_brian_2_prototype36.py`

SHA-256: `acaea2170c04c274bc4cc5d50db69a6a316f4b3f2c99ff80878b9cdddd6ed0d3`

Сравнение с `rings_minigrid_brian_2_prototype35.py`.

Функции: `Δ sample_params`.

Параметры по умолчанию:

- `w_danger_ahead_forward_inh`: `10.0` → `20.0`
- `w_danger_ahead_to_turn`: `8.0` → `12.0`
- `w_escape_to_motor_forward`: `7.0` → `3.5`
- `w_motor_self_support`: `1.8` → `0.8`

### Файл `rings_minigrid_brian_2_prototype37.py`

SHA-256: `c064e2aecd62a3e03e027909b045acc62c95fb20fde710cbf0e05e1b312f0bda`

Сравнение с `rings_minigrid_brian_2_prototype36.py`.

Функции: `+ Danger.encode`; `+ Food.encode`; `Δ RingsSNNController.__init__`; `+ _init_tune_worker`; `+ _save_tune_payload`; `+ _tune_trial`; `Δ evaluate_params`; `Δ main`; `Δ run_episode`; `Δ tune_random_search`.
