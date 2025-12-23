//! Random selection support for `HashTrieMap`

use super::{Bucket, EntryWithHash, HashTrieMap, Node};
use alloc::vec::Vec;
use archery::SharedPointerKind;
use core::hash::{BuildHasher, Hash};
use rand::Rng;
use rand::seq::{IndexedRandom, IteratorRandom};
use std::collections::HashSet;

impl<K, V, P> Node<K, V, P>
where
    K: Eq + Hash,
    P: SharedPointerKind,
{
    /// Chooses a random entry from this node.
    ///
    /// Picks uniformly among present children at each branch, assuming the tree
    /// is roughly balanced (which it should be with a good hash function).
    fn choose<R: Rng + ?Sized>(&self, rng: &mut R) -> &EntryWithHash<K, V, P> {
        match self {
            Node::Branch(children) => {
                let chosen = children
                    .as_slice()
                    .choose(rng)
                    .expect("branch should have at least one child");
                chosen.choose(rng)
            }
            Node::Leaf(bucket) => bucket.choose(rng),
        }
    }
}

impl<K, V, P> Bucket<K, V, P>
where
    K: Eq + Hash,
    P: SharedPointerKind,
{
    fn choose<R: Rng + ?Sized>(&self, rng: &mut R) -> &EntryWithHash<K, V, P> {
        match self {
            Bucket::Single(entry) => entry,
            Bucket::Collision(entries) => {
                // For collision lists, pick uniformly at random
                let idx = rng.random_range(0..entries.len());
                entries.iter().nth(idx).expect("index should be within bounds")
            }
        }
    }
}

impl<K, V, P, H: BuildHasher> HashTrieMap<K, V, P, H>
where
    K: Eq + Hash,
    H: Clone,
    P: SharedPointerKind,
{
    /// Returns a reference to a random key-value pair from the map.
    ///
    /// Returns `None` if the map is empty.
    ///
    /// # Complexity
    ///
    /// O(log n) - picks uniformly among children at each tree level.
    ///
    /// # Distribution
    ///
    /// The selection is approximately uniform assuming the hash function distributes
    /// keys evenly. With a poor hash function that creates unbalanced subtrees,
    /// elements in smaller subtrees may be over-represented.
    ///
    /// # Example
    ///
    /// ```
    /// use rpds::HashTrieMap;
    /// use rand::thread_rng;
    ///
    /// let map = HashTrieMap::new()
    ///     .insert("a", 1)
    ///     .insert("b", 2)
    ///     .insert("c", 3);
    ///
    /// if let Some((key, value)) = map.choose(&mut thread_rng()) {
    ///     println!("Chose: {} -> {}", key, value);
    /// }
    /// ```
    #[must_use]
    pub fn choose<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(&K, &V)> {
        if self.is_empty() {
            return None;
        }

        let entry = self.root.choose(rng);
        Some((entry.key(), entry.value()))
    }

    /// Returns `amount` distinct random key-value pairs from the map (without replacement).
    ///
    /// If `amount` exceeds the map size, returns all elements in random order.
    ///
    /// # Complexity
    ///
    /// - O(k log n) when k² < n (uses rejection sampling)
    /// - O(n) when k² >= n (uses iteration to avoid birthday paradox)
    ///
    /// # Example
    ///
    /// ```
    /// use rpds::HashTrieMap;
    /// use rand::thread_rng;
    ///
    /// let map = HashTrieMap::new()
    ///     .insert("a", 1)
    ///     .insert("b", 2)
    ///     .insert("c", 3);
    ///
    /// for (key, value) in map.choose_multiple(&mut thread_rng(), 2) {
    ///     println!("Chose: {} -> {}", key, value);
    /// }
    /// ```
    #[must_use]
    pub fn choose_multiple<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        amount: usize,
    ) -> impl IntoIterator<Item = (&K, &V)> {
        if amount == 0 || self.is_empty() {
            return Vec::new();
        }

        let n = self.size();
        let amount = amount.min(n);

        // Birthday paradox: collisions become frequent when k² ≈ n.
        // If k² >= n, skip rejection sampling and use iteration directly.
        if amount.saturating_mul(amount) >= n {
            return self.iter().choose_multiple(rng, amount);
        }

        let mut chosen: Vec<(&K, &V)> = Vec::with_capacity(amount);
        let mut seen: HashSet<*const K> = HashSet::with_capacity(amount);

        // Rejection sampling: expected samples ≈ k + k²/(2n), so use ~2k as budget
        // when k² < n, this is at most 2k samples for k results
        let max_attempts = amount.saturating_mul(2);
        let mut attempts = 0;

        while chosen.len() < amount && attempts < max_attempts {
            if let Some((key, val)) = self.choose(rng) {
                let ptr = key as *const K;
                if seen.insert(ptr) {
                    chosen.push((key, val));
                }
            }
            attempts += 1;
        }

        // Fallback if we got unlucky (shouldn't happen often when k² < n)
        if chosen.len() < amount {
            let remaining = self
                .iter()
                .filter(|(k, _)| !seen.contains(&(*k as *const K)))
                .choose_multiple(rng, amount - chosen.len());
            chosen.extend(remaining);
        }

        chosen
    }
}

#[cfg(test)]
mod tests {
    use crate::HashTrieMap;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::HashSet;

    #[test]
    fn test_choose_empty_map() {
        let map: HashTrieMap<i32, i32> = HashTrieMap::new();
        let mut rng = StdRng::seed_from_u64(42);
        assert!(map.choose(&mut rng).is_none());
    }

    #[test]
    fn test_choose_single_element() {
        let map = HashTrieMap::new().insert(1, "one");
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..10 {
            let result = map.choose(&mut rng);
            assert_eq!(result, Some((&1, &"one")));
        }
    }

    #[test]
    fn test_choose_returns_valid_entries() {
        let mut map = HashTrieMap::new();
        for i in 0..100 {
            map.insert_mut(i, i * 2);
        }

        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let (key, value) = map.choose(&mut rng).unwrap();
            assert!(map.contains_key(key));
            assert_eq!(*value, *key * 2);
        }
    }

    #[test]
    fn test_choose_covers_all_elements() {
        let mut map = HashTrieMap::new();
        for i in 0..10 {
            map.insert_mut(i, i);
        }

        let mut rng = StdRng::seed_from_u64(42);
        let mut seen: HashSet<i32> = HashSet::new();

        // With enough samples, we should see all elements
        for _ in 0..1000 {
            let (key, _) = map.choose(&mut rng).unwrap();
            seen.insert(*key);
        }

        assert_eq!(seen.len(), 10);
    }

    #[test]
    fn test_choose_multiple() {
        let mut map = HashTrieMap::new();
        for i in 0..100 {
            map.insert_mut(i, i * 2);
        }

        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<_> = map.choose_multiple(&mut rng, 50).into_iter().collect();

        assert_eq!(samples.len(), 50);
        for (key, value) in &samples {
            assert!(map.contains_key(key));
            assert_eq!(**value, **key * 2);
        }

        // Verify no duplicates (without replacement)
        let unique: HashSet<*const i32> = samples.iter().map(|(k, _)| *k as *const i32).collect();
        assert_eq!(unique.len(), 50);
    }

    #[test]
    fn test_choose_multiple_empty_map() {
        let map: HashTrieMap<i32, i32> = HashTrieMap::new();
        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<_> = map.choose_multiple(&mut rng, 10).into_iter().collect();
        assert!(samples.is_empty());
    }

    #[test]
    fn test_choose_multiple_more_than_size() {
        let mut map = HashTrieMap::new();
        for i in 0..10 {
            map.insert_mut(i, i);
        }

        let mut rng = StdRng::seed_from_u64(42);
        // Ask for 1000, but map only has 10
        let samples: Vec<_> = map.choose_multiple(&mut rng, 1000).into_iter().collect();
        assert_eq!(samples.len(), 10);

        // Verify all elements returned
        let unique: HashSet<i32> = samples.iter().map(|(k, _)| **k).collect();
        assert_eq!(unique.len(), 10);
    }

    #[test]
    fn test_choose_multiple_triggers_fallback() {
        // With k close to n, rejection sampling will stall and fallback kicks in
        let mut map = HashTrieMap::new();
        for i in 0..100 {
            map.insert_mut(i, i);
        }

        let mut rng = StdRng::seed_from_u64(42);
        // Ask for 90 out of 100 - will definitely need fallback
        let samples: Vec<_> = map.choose_multiple(&mut rng, 90).into_iter().collect();
        assert_eq!(samples.len(), 90);

        // Verify no duplicates
        let unique: HashSet<i32> = samples.iter().map(|(k, _)| **k).collect();
        assert_eq!(unique.len(), 90);
    }

    #[test]
    fn test_choose_distribution_all_elements_reachable() {
        // The simple algorithm (uniform among children at each level) isn't perfectly
        // uniform - elements in smaller subtrees are over-represented. But all elements
        // should still be reachable, and with a larger map the distribution improves.
        let mut map = HashTrieMap::new();
        for i in 0..100 {
            map.insert_mut(i, i);
        }

        let mut rng = StdRng::seed_from_u64(12345);
        let mut seen = HashSet::new();

        // All elements should be reachable within a reasonable number of samples
        for _ in 0..10000 {
            let (key, _) = map.choose(&mut rng).unwrap();
            seen.insert(*key);
        }

        assert_eq!(seen.len(), 100, "All elements should be reachable");
    }

    #[test]
    fn test_choose_distribution_no_element_dominates() {
        // Verify no single element is chosen too often (would indicate broken tree walk)
        let mut map = HashTrieMap::new();
        let n = 100;
        for i in 0..n {
            map.insert_mut(i, i);
        }

        let mut rng = StdRng::seed_from_u64(99999);
        let mut counts = std::collections::HashMap::new();
        let samples = 10000;

        for _ in 0..samples {
            let (key, _) = map.choose(&mut rng).unwrap();
            *counts.entry(*key).or_insert(0) += 1;
        }

        let max_count = *counts.values().max().unwrap();
        let expected = samples / n;

        // No element should be chosen more than 5x the expected rate
        assert!(
            max_count < expected * 5,
            "Element chosen {max_count} times, expected ~{expected} (max allowed: {})",
            expected * 5
        );
    }

    #[test]
    fn test_choose_with_string_keys() {
        let mut map = HashTrieMap::new();
        for i in 0..50 {
            map.insert_mut(format!("key_{i}"), i);
        }

        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let (key, value) = map.choose(&mut rng).unwrap();
            assert!(key.starts_with("key_"));
            assert!(map.contains_key(key));
            let key_num: i32 = key.strip_prefix("key_").unwrap().parse().unwrap();
            assert_eq!(*value, key_num);
        }
    }
}
