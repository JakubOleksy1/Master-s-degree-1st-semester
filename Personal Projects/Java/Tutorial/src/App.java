/* TUTORIAL 21
class Cat {
    private String name;
    private int age;

    public void setName(String name) {
        this.name = name;
    }

    public void setAge(int age) {
        if(age > 0) {
            this.age = age;
        } else {
            System.out.println("Age must be greater than 0");
        }
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }
}

public class App {
    public static void main(String[] args) {
        Cat cloey = new Cat();
        // frog1.name = "Bertie";
        // frog1.age = 1;
        cloey.setName("Cloey");
        cloey.setAge(1);

        System.out.println(cloey.getName());
        System.out.println(cloey.getAge());
    }
}*/

/* TUTORIAL 20 
class Robot {
    public static void main(String[] args) {
        System.out.println("Hel;o");
    }
    
    public void speak(String text) {
        System.out.println(text);
    }

    public void jump(int height) {
        System.out.println("Jumping: " + height);
    }

    public void move(String direction, double distance) {
        System.out.println("Moving " + distance + " meters in direction " + direction);
    }
}

public class App {
    public static void main(String[] args) {
        Robot robot = new Robot();
        robot.speak("HI I am a robot.");
        robot.jump(7);
        robot.move("West", 12.2);
        
        String greeting = "Hello there!";
        robot.speak(greeting);

        int value = 14;
        robot.jump(value);
    }
}*/

/* TUTORIAL 19
class Person {
    String name;
    int age;

    void speak() {
        System.out.println("My name is: " + name + " and I am " + age + " years old.");
    }

    int calculateYearsToRetirement() {
        int yearsLeft = 65 - age;
        System.out.println(yearsLeft);

        return yearsLeft;
    }

    int getAge() {
        return age;
    }

    String getName() {
        return name;
    }
}

public class App { 
    public static void main(String[] args) {
        Person person1 = new Person();
        
        person1.name = "Joe ";
        person1.age = 25;
        
        // person1.speak();

        int years = person1.calculateYearsToRetirement();

        System.out.println("Years until retirement: " + years);
    
        int age = person1.getAge();
        System.out.println("Age is: " + age);

        String name = person1.getName();
        System.out.println("Name is: " + name);
    }
}*/

/*TUTORIAL 17 & 18
class Person {
    String name;
    int age;

    void speak() {
        for(int i = 0; i < 3; i++) {
            System.out.println("My name is: " + name + " and I am " + age + " years old.");
        }
    }

    void sayHello() {
        System.out.println("Hello there!");
    }
}

public class App { 
    public static void main(String[] args) {
        Person person1 = new Person();
        person1.name = "Joe Bloggs";
        person1.age = 37;
        person1.speak();
        person1.sayHello();

        Person person2 = new Person();
        person2.name = "Sarah Smith";
        person2.age = 20;
        person2.speak();
        person2.sayHello();

        System.out.println(person1.name);
    }
}*/

/*TUTORIAL 16
public class App {
    public static void main(String[] args) {
        int[] values = {3, 5, 2343};
        System.out.println(values[2]);

        int[][] grid = {
            {3, 5, 2343},
            {2, 4},
            {1, 2, 3, 4}
        };

        System.out.println(grid[1][1]);
        System.out.println(grid[0][2]);
        
        String[][] texts = new String[2][3];
        texts[0][1] = "Hello there";

        System.out.println(texts[0][1]);

        for(int row = 0; row < grid.length; row++) {
            for(int col = 0; col < grid[row].length; col++) {
                System.out.print(grid[row][col] + "\t");
            }
            System.out.println();
        }

        String[][] words = new String[2][];

        System.out.println(words[0]);

        words[0] = new String[3];

        words[0][1] = "Hi there!";
        System.out.println(words[0][1]);
    }
}*/

/*TUTORIAL 15
public class App {
    public static void main(String[] args) {
        String[] words = new String[3];
        words[0] = "Hello";
        words[1] = "to";
        words[2] = "you";

        System.out.println(words[2]);

        String[] fruits = {"apple", "banana", "pear", "kiwi"};

        for(String fruit: fruits) {
            System.out.println(fruit);
        }

        int value = 0;
        String text = null;
        System.out.println(text);
        
        String[] texts = new String[2];
        System.out.println(texts[value]);

        texts[0] = "one";
    }
}*/

/*TUTORIAL 14 
public class App {
    public static void main(String[] args) {

        int value = 0;
        int[] values;
        values = new int[3];
        System.out.println(values[0]);

        values[0] = 10;
        values[1] = 20;
        values[2] = 30;

        System.out.println(values[0]);
        System.out.println(values[1]);
        System.out.println(values[2]);

        for(int i = 0; i < values.length; i++) {
            System.out.println(values[i]);
        }

        int[] numbers = {5, 6, 7};

        for(int i = 0; i < numbers.length; i++) {
            System.out.println(numbers[i]);
        }
    }
}*/

/* TUTORIAL 13
import java.util.Scanner;

public class App {
    public static void main(String[] args) {
        
        Scanner input = new Scanner(System.in);

        System.out.println("Enter a command: ");
        String text = input.nextLine();
        input.close();
        switch (text) {
            case "start":
                System.out.println("Machine started!");
                break;
            
                case "stop":
                System.out.println("Machine stopped!");
                break;
            
                default:
                System.out.println("Command not recognized!");
                break;
        }
    }
}*/

/*TUTORIAL 12
import java.util.Scanner;
public class App {
    public static void main(String[] args) {
        
        Scanner scanner = new Scanner(System.in);

        /*System.out.println("Enter a number: ");

        int value = scanner.nextInt();
        while(value != 5) {
            System.out.println("Enter a number: ");
            value = scanner.nextInt();
        }
        int value = 0;
        do {
            System.out.println("Enter a number: ");
            value = scanner.nextInt();
        } while(value != 5);

        System.out.println("You got 5!");
        scanner.close();
    }
}*/

/*TUTORIAL 11
import java.util.Scanner;

public class App {
    public static void main(String[] args) {
        
        // Create scanner object
        Scanner input = new Scanner(System.in);
        
        // Output the prompt
        System.out.println("Enter an intiger: ");
        
        // Wait for the user to enter a line of text
        int value = input.nextInt();

        // Tell them what they entered
        System.out.println("You entered: " + value);
    }
}*/

/*TUTORIAL 6 7 8 9 10
public class App {
    public static void main(String[] args) {
        int loop = 0;

        while(true) {
            System.out.println("Looping: " + loop);
            if(loop == 5) {
                break;
            }
            loop++;
            System.out.println("Running");
        }
    }
}*/

/*TUTORIAL 5 
public class App {
    public static void main(String[] args) {
        for(int i = 0; i<5; i++) {
            System.out.printf("The value of i is: %d\n", i);
        }
    } 
}*/

/*TUTORIAL 4 
public class App {
    public static void main(String[] args) {
        int value = 0;
        while(value < 10) {
            System.out.println("Hello " + value);
            value = value + 1;
        }
    } 
}*/

/*TUTORIAL 3 
public class App {
    public static void main(String[] args) {
        int myInt = 7;

        String text = "Hello";
        String blank = " ";
        String name = "Bob";
        String greeting = text + blank + name;
        System.out.println(greeting);
        System.out.println("Hello" + " " + "Bob");
        System.out.println("My integer is: " + myInt);
        double myDouble = 7.8;
        System.out.println("My number is: " + myDouble + ".");
    } 
}*/

/*TUTORIAL 1 & 2 
public class App {
    public static void main(String[] args) throws Exception {
        System.out.println("Hello, World!");
        int myNumber = 88;
        short myShort = 847;
        long myLong = 9797;
        double myDouble = 7.3243;
        float myFloat = 324.3f;
        char myChar = 'y';
        boolean myBoolean = true;
        byte myByte = 127;

        System.out.println(myNumber);
        System.out.println(myShort);
        System.out.println(myLong);
        System.out.println(myDouble);
        System.out.println(myFloat);
        System.out.println(myChar);
        System.out.println(myBoolean);
        System.out.println(myByte);
    }
}*/
