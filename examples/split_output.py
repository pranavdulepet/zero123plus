from PIL import Image

input_image_path = './test_output_birdanimal1.png'
main_image = Image.open(input_image_path)

num_columns = 2
num_rows = 3
sub_image_width = main_image.width // num_columns
sub_image_height = main_image.height // num_rows

for row in range(num_rows):
    for col in range(num_columns):
        left = col * sub_image_width
        upper = row * sub_image_height
        right = left + sub_image_width
        lower = upper + sub_image_height

        sub_image = main_image.crop((left, upper, right, lower))

        sub_image_path = f'./split_output/sub_image_{row * num_columns + col + 1}.png'
        sub_image.save(sub_image_path)
        print(f'Saved sub-image to: {sub_image_path}')

