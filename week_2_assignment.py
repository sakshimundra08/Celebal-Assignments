class Node:
    def __init__(self, data):
        self.data = data 
        self.next = None  

class LinkedList:
    def __init__(self):
        self.head = None  

    def add_to_end(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        if self.head is None:
            print("The list is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        if self.head is None:
            print("Can't delete from an empty list!")
            return
        if n <= 0:
            print("Please enter a valid position (1 or greater).")
            return
        if n == 1:
            self.head = self.head.next
            return
        current = self.head
        for i in range(n - 2):
            if current.next is None:
                print("That position is out of range.")
                return
            current = current.next
        if current.next is None:
            print("That position is out of range.")
            return
        current.next = current.next.next

if __name__ == "__main__":
    my_list = LinkedList()

    my_list.add_to_end(5)
    my_list.add_to_end(15)
    my_list.add_to_end(25)
    my_list.add_to_end(35)

    print("Here is the list for you:")
    my_list.print_list()

    print("\nDeleting node 3...")
    my_list.delete_nth_node(3)
    my_list.print_list()

    print("\nTrying to delete a node that doesn't exist (at position 10):")
    my_list.delete_nth_node(10)

    print("\nDeleting all the nodes one by one:")
    my_list.delete_nth_node(1)
    my_list.delete_nth_node(1)
    my_list.delete_nth_node(1)
    my_list.print_list()

    print("\nTrying to delete from an empty list:")
    my_list.delete_nth_node(1)
