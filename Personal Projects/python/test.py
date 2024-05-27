class Solution:
    def romanToInt(self, s: str) -> int:
        roman_values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        
        total = 0
        prev_value = 0
        
        for char in s:
            current_value = roman_values[char]
            
            if current_value > prev_value:
                # If the current value is greater than the previous one,
                # subtract the previous value twice (once for the previous
                # iteration, and once for the current iteration).
                total += current_value - 2 * prev_value
            else:
                total += current_value
                
            prev_value = current_value
        
        return total

# Example usage:
sol = Solution()
print(sol.romanToInt("III"))       # Output: 3
print(sol.romanToInt("LVIII"))     # Output: 58
print(sol.romanToInt("MCMXCIV"))   # Output: 1994