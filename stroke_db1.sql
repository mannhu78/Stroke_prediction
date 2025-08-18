-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Máy chủ: 127.0.0.1
-- Thời gian đã tạo: Th4 30, 2025 lúc 08:52 AM
-- Phiên bản máy phục vụ: 10.4.32-MariaDB
-- Phiên bản PHP: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Cơ sở dữ liệu: `stroke_db1`
--

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `prediction`
--

CREATE TABLE `prediction` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `gender` float DEFAULT NULL,
  `age` float DEFAULT NULL,
  `hypertension` float DEFAULT NULL,
  `heart_disease` float DEFAULT NULL,
  `ever_married` float DEFAULT NULL,
  `work_type` float DEFAULT NULL,
  `Residence_type` float DEFAULT NULL,
  `avg_glucose_level` float DEFAULT NULL,
  `bmi` float DEFAULT NULL,
  `smoking_status` float DEFAULT NULL,
  `result` float DEFAULT NULL,
  `timestamp` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Đang đổ dữ liệu cho bảng `prediction`
--

INSERT INTO `prediction` (`id`, `user_id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `result`, `timestamp`) VALUES
(1, 2, 0, 20, 0, 0, 0, 0, 1, 120, 18.6, 2, 0, '2025-04-12 23:13:49'),
(2, 2, 1, 50, 1, 1, 1, 1, 1, 228.69, 36.6, 1, 18, '2025-04-12 23:15:57'),
(3, 2, 0, 50, 1, 1, 1, 1, 1, 200, 36.6, 2, 17, '2025-04-12 23:36:48'),
(4, 2, 1, 50, 1, 1, 1, 1, 1, 200, 36.6, 1, 18, '2025-04-12 23:38:56'),
(5, 2, 1, 50, 1, 1, 1, 2, 1, 200, 36.6, 1, 19, '2025-04-12 23:40:24'),
(6, 3, 1, 60, 1, 1, 1, 4, 1, 300, 38.6, 1, 26, '2025-04-13 11:43:09'),
(7, 3, 0, 45, 1, 1, 1, 3, 0, 202.1, 34.4, 2, 15, '2025-04-13 11:56:49'),
(8, 3, 1, 27, 0, 0, 0, 4, 1, 234.58, 34.3, 3, 0, '2025-04-13 11:58:59'),
(9, 2, 1, 40, 1, 0, 1, 4, 1, 200, 35.5, 1, 1, '2025-04-14 15:14:37'),
(10, 2, 1, 40, 1, 0, 1, 4, 1, 200, 35.5, 1, 1, '2025-04-14 15:16:07'),
(11, 4, 1, 21, 0, 0, 0, 0, 1, 90, 18, 1, 0, '2025-04-14 20:55:20'),
(12, 5, 0, 20, 0, 0, 0, 0, 1, 20, 35.5, 2, 0, '2025-04-23 22:10:40'),
(13, 5, 1, 25, 1, 0, 0, 0, 1, 25, 35.5, 2, 6, '2025-04-24 23:33:14'),
(14, 5, 1, 30, 0, 1, 0, 1, 1, 20, 35.5, 1, 5, '2025-04-24 23:36:47'),
(15, 5, 0, 21, 0, 0, 0, 0, 1, 20, 17.5, 2, 0, '2025-04-26 17:39:45'),
(16, 5, 0, 20, 0, 1, 0, 0, 1, 20, 17.5, 2, 8, '2025-04-26 17:57:04'),
(17, 5, 1, 35, 0, 0, 0, 1, 1, 135, 19.1, 1, 0, '2025-04-28 21:38:22'),
(18, 5, 0, 45, 1, 1, 1, 4, 0, 120, 19.5, 2, 10, '2025-04-28 22:11:20');

-- --------------------------------------------------------

--
-- Cấu trúc bảng cho bảng `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Đang đổ dữ liệu cho bảng `users`
--

INSERT INTO `users` (`id`, `email`, `password`) VALUES
(1, 'admin@example.com', '$pbkdf2-sha256$29000$ytC3v4VWStkYmF9P4Z4e3A$5AE0j9FZ6MyM63VMygYHGH/tUZo5hBPuzN6tLdjPGB0'),
(2, 'mannhu@gmail.com', 'pbkdf2:sha256:1000000$oCpWMaQN9Nf9nAht$4d9f2c1bc3761f66e67cfeb92d413cdb74dc777ee0b5564d4a2950bf05c061b2'),
(3, 'abc@gmail.com', 'pbkdf2:sha256:1000000$J83ojwd5ej7C0dDx$9b6d210ff059aae733482980d7bb7f4a40dc990455d718e3de1e9eef19172a5d'),
(4, 'phuthanh@gmail.com', 'pbkdf2:sha256:1000000$l8CxgEWVdZY1LgHI$e94c9e52e943354bb1ddd8b538860aff12fb99f6881a96b254bc7947159bde44'),
(5, 'admin@gmail.com', 'pbkdf2:sha256:1000000$Ia8dCHufwyLJpb7d$fde0d5f3c927e559e8d9711a0e8eb1feabe88b8317b450917daaa2041304fc91');

--
-- Chỉ mục cho các bảng đã đổ
--

--
-- Chỉ mục cho bảng `prediction`
--
ALTER TABLE `prediction`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Chỉ mục cho bảng `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT cho các bảng đã đổ
--

--
-- AUTO_INCREMENT cho bảng `prediction`
--
ALTER TABLE `prediction`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=19;

--
-- AUTO_INCREMENT cho bảng `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- Các ràng buộc cho các bảng đã đổ
--

--
-- Các ràng buộc cho bảng `prediction`
--
ALTER TABLE `prediction`
  ADD CONSTRAINT `prediction_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
