class Solution:
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(numbers)
        left, right = 0, n-1
        while left < right:
            if numbers[left]+numbers[right]==target: return [left+1, right+1]
            if numbers[left]+numbers[right] < target: left += 1
            if numbers[left]+numbers[right] > target: right -= 1

class Solution:
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(numbers)
        i, j = 0, n-1
        res = [0, 0]
        while i < j:
            cur_sum = numbers[i]+numbers[j]
            if cur_sum == target: return [i+1, j+1]
            elif cur_sum > target: j -= 1
            else: i += 1
        return res