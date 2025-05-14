#!/bin/bash

git fetch --all

cd ~/Desktop/diploma-similarity-analyzer || {
    echo "❌ Papkaga o'tib bo'lmadi. Yo'lni tekshiring."
    exit 1
}

# Branchlar ro'yxatini olish
branches=($(git branch -r | grep -v '\->' | sed 's/origin\///'))

# --- Master branchni tekshirish ---
if ! printf '%s\n' "${branches[@]}" | grep -q '^master$'; then
    echo "⚠️  Uzoqdagi 'master' branch topilmadi."
    read -p "Yangi master branch yaratilsinmi? (ha/yo'q): " create_master
    if [[ "$create_master" == "ha" ]]; then
        git checkout -b master || {
            echo "❌ master branchni yaratib bo'lmadi."
            exit 1
        }
        git push -u origin master || {
            echo "❌ master branchni remote ga push qilib bo'lmadi."
            exit 1
        }
        echo "✅ master branch yaratildi va GitHub'ga yuklandi."
        branches+=("master")
    fi
else
    # Lokal master mavjud emas, lekin origin/master bor bo‘lsa, ko‘chirib olish
    if ! git branch | grep -q 'master'; then
        git checkout -b master origin/master
    fi
fi

echo "=== GitHub branchlar ro'yxati ==="
for i in "${!branches[@]}"; do
    echo "$i) ${branches[$i]}"
done

# Foydalanuvchidan tanlov
read -p "Branch raqamini tanlang (masalan, 0): " choice

# Tanlovni tekshirish
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -ge "${#branches[@]}" ]; then
    echo "❌ Noto'g'ri tanlov. Dastur yakunlandi."
    exit 1
fi

branch=${branches[$choice]}
echo "📦 Tanlangan branch: $branch"

# Branchga o'tish
git checkout "$branch" || {
    echo "❌ Branchga o'tib bo'lmadi."
    exit 1
}

# Git add va commit
git add .

git commit -m "Avtomatik push: $(date)" || {
    echo "⚠️  Hech qanday o'zgarish yo'q yoki commitda xato yuz berdi."
    exit 1
}

# Push
git push origin "$branch" || {
    echo "❌ Push qilishda xato yuz berdi. Autentifikatsiyani tekshiring yoki git pull qiling."
    exit 1
}

echo "✅ Barcha o'zgarishlar GitHub branch '$branch' ga muvaffaqiyatli yuklandi!"
