import sqlite3

conn = sqlite3.connect("college.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    department_id INTEGER,
    level TEXT NOT NULL, -- UG, PG, Ph.D., Diploma
    duration TEXT NOT NULL,
    seats INTEGER NOT NULL,
    fees INTEGER NOT NULL,
    eligibility TEXT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
)
''')

cursor.executemany('''
INSERT INTO departments (name) VALUES (?)
''', [
    ("Computer Science",),
    ("Electronics and Communication",),
    ("Mechanical Engineering",),
    ("Civil Engineering",),
    ("Biotechnology",),
    ("Business Administration",),
    ("Commerce",),
    ("Physics",),
    ("Chemistry",),
    ("Mathematics",),
    ("English",),
    ("Psychology",),
    ("Hotel Management",),
    ("Pharmacy",),
    ("Nursing",)
])

cursor.executemany('''
INSERT INTO courses (name, department_id, level, duration, seats, fees, eligibility) VALUES (?, ?, ?, ?, ?, ?, ?)
''', [

    ("B.Sc. Computer Science", 1, "UG", "3 Years", 60, 75000, "12th Grade with Maths"),
    ("B.Tech Computer Engineering", 1, "UG", "4 Years", 60, 120000, "12th Grade with PCM"),
    ("B.Sc. Electronics", 2, "UG", "3 Years", 50, 70000, "12th Grade with Science"),
    ("B.Tech Mechanical Engineering", 3, "UG", "4 Years", 50, 110000, "12th Grade with PCM"),
    ("B.Tech Civil Engineering", 4, "UG", "4 Years", 50, 105000, "12th Grade with PCM"),
    ("B.Sc. Biotechnology", 5, "UG", "3 Years", 40, 80000, "12th Grade with Science"),
    ("BBA", 6, "UG", "3 Years", 60, 90000, "12th Grade"),
    ("B.Com", 7, "UG", "3 Years", 60, 85000, "12th Grade"),
    ("B.Sc. Physics", 8, "UG", "3 Years", 40, 70000, "12th Grade with Science"),
    ("B.Sc. Chemistry", 9, "UG", "3 Years", 40, 72000, "12th Grade with Science"),
    ("B.Sc. Mathematics", 10, "UG", "3 Years", 40, 68000, "12th Grade with Maths"),
    ("B.A. English", 11, "UG", "3 Years", 50, 65000, "12th Grade"),
    ("B.A. Psychology", 12, "UG", "3 Years", 50, 70000, "12th Grade"),
    ("B.Sc. Hotel Management", 13, "UG", "3 Years", 50, 75000, "12th Grade"),
    ("B.Pharm", 14, "UG", "4 Years", 50, 95000, "12th Grade with Science"),


    ("M.Sc. Computer Science", 1, "PG", "2 Years", 30, 120000, "UG in Computer Science"),
    ("M.Tech Electronics", 2, "PG", "2 Years", 25, 130000, "UG in Electronics"),
    ("MBA", 6, "PG", "2 Years", 50, 140000, "UG in any field"),
    ("M.Com", 7, "PG", "2 Years", 40, 125000, "UG in Commerce"),
    ("M.Sc. Mathematics", 10, "PG", "2 Years", 30, 115000, "UG in Mathematics"),
    ("M.A. English", 11, "PG", "2 Years", 30, 110000, "UG in English"),
    ("M.Sc. Psychology", 12, "PG", "2 Years", 30, 120000, "UG in Psychology"),

    ("Ph.D. Computer Science", 1, "Ph.D.", "3-5 Years", 10, 180000, "PG in relevant field"),
    ("Ph.D. Electronics", 2, "Ph.D.", "3-5 Years", 8, 185000, "PG in relevant field"),
    ("Ph.D. Mechanical Engineering", 3, "Ph.D.", "3-5 Years", 6, 190000, "PG in relevant field"),
    ("Ph.D. Biotechnology", 5, "Ph.D.", "3-5 Years", 8, 200000, "PG in relevant field"),
    ("Ph.D. Physics", 8, "Ph.D.", "3-5 Years", 7, 175000, "PG in relevant field"),
    ("Ph.D. Chemistry", 9, "Ph.D.", "3-5 Years", 7, 180000, "PG in relevant field"),
    ("Ph.D. Psychology", 12, "Ph.D.", "3-5 Years", 5, 185000, "PG in relevant field"),

    ("Diploma in Computer Applications", 1, "Diploma", "1 Year", 40, 45000, "10th or 12th Grade"),
    ("Diploma in Hotel Management", 13, "Diploma", "1 Year", 40, 50000, "10th or 12th Grade"),
    ("Diploma in Business Administration", 6, "Diploma", "1 Year", 40, 48000, "10th or 12th Grade")
])

conn.commit()
conn.close()

print("✅ Database created, tables initialized, and sample data inserted!")
